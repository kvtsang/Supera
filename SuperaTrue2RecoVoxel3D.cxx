#ifndef __SuperaTrue2RecoVoxel3D_CXX__
#define __SuperaTrue2RecoVoxel3D_CXX__

#include <unordered_set>
#include <fstream>
#include <algorithm>
#include "SuperaTrue2RecoVoxel3D.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"

namespace larcv 
{
  static SuperaTrue2RecoVoxel3DProcessFactory __global_SuperaTrue2RecoVoxel3DProcessFactory__;

  SuperaTrue2RecoVoxel3D::SuperaTrue2RecoVoxel3D(const std::string name)
    : SuperaBase(name)
  {}

  void SuperaTrue2RecoVoxel3D::configure(const PSet& cfg)
  {
    SuperaBase::configure(cfg);
    _output_tensor3d = cfg.get<std::string>("OutputTensor3D","");
    _output_cluster3d = cfg.get<std::string>("OutputCluster3D","");
    _debug = cfg.get<bool> ("DebugMode",false);
    _hit_producer = cfg.get<std::string>("LArHitProducer","gaushit");
    _sps_producer = cfg.get<std::string>("LArSpacePointProducer","cluster3d");
    _use_true_pos = cfg.get<bool>("UseTruePosition",true);
    _twofold_matching = cfg.get<bool>("TwofoldMatching", false);
    _ref_meta3d_cluster3d = cfg.get<std::string>("Meta3DFromCluster3D","pcluster");
    _ref_meta3d_tensor3d = cfg.get<std::string>("Meta3DFromTensor3D","");
    _hit_threshold_ne = cfg.get<double>("HitThresholdNe", 0);
    _hit_window_ticks = cfg.get<double>("HitWindowTicks", 5.);
    _hit_peak_finding = cfg.get<bool>("HitPeakFinding", false);
    _dump_to_csv = cfg.get<bool>("DumpToCSV", false);
    _post_averaging = cfg.get<bool>("PostAveraging", false);
    _post_averaging_threshold = cfg.get<double>("PostAveragingThreshold_cm", 0.3);
  }


  void SuperaTrue2RecoVoxel3D::initialize()
  {
    SuperaBase::initialize();

    // dump channel map
    if (_dump_to_csv) {
      art::ServiceHandle<geo::Geometry const> geom;

      std::ofstream out("channel_map.csv");
      out << "ch,cryo,tpc,plane,wire\n";

      for (size_t ch = 0; ch < geom->Nchannels(); ++ch) {

        auto const& ids = geom->ChannelToWire(ch);
        for (auto const& id : ids) {
          out << ch << ','
            << id.Cryostat << ','
            << id.TPC << ','
            << id.Plane << ','
            << id.Wire << '\n';
        }
      }
      out.close();
    }
  }

  larcv::Voxel3DMeta SuperaTrue2RecoVoxel3D::get_meta3d(IOManager& mgr) const {

    larcv::Voxel3DMeta meta3d;
    if(!_ref_meta3d_cluster3d.empty()) {
      auto const& ev_cluster3d = mgr.get_data<larcv::EventClusterVoxel3D>(_ref_meta3d_cluster3d);
      meta3d = ev_cluster3d.meta();
    }
    else if(!_ref_meta3d_tensor3d.empty()) {
      auto const& ev_tensor3d = mgr.get_data<larcv::EventSparseTensor3D>(_ref_meta3d_tensor3d);
      meta3d = ev_tensor3d.meta();
    }
    else
      LARCV_CRITICAL() << "ref_meta3d_cluster3d nor ref_meta3d_tensor3d set!" << std::endl;
    return meta3d;
  }

  bool SuperaTrue2RecoVoxel3D::process(IOManager& mgr)
  {
    SuperaBase::process(mgr);
		auto event_id = mgr.event_id().event();

   // clear true2reco, ghosts maps
    clear_maps();

    // setup csv logger
    std::map<std::string, std::ofstream> csv;
    if (_dump_to_csv) {
      auto save_to = [&](std::string prefix)
      {
        return prefix + "_" + std::to_string(event_id) + ".csv";
      };

      std::vector<std::tuple<std::string, std::string>> tables;
      tables.emplace_back("simch", "ch,time,track_id,n_e,energy,x,y,z,ix,iy,iz,id");
      tables.emplace_back("gaushits", "ch,time,rms,amp,charge");
      tables.emplace_back("cluster3d", "id,x,y,z,charge,ch1,t1,rms1,ch2,t2,rms2,ch3,t3,rms3");
      tables.emplace_back("true3d", "id,x,y,z");
      tables.emplace_back("reco3d", "id,x,y,z");
      tables.emplace_back("ghosts", "id,is_ghost");
      tables.emplace_back("true2reco", "true_id,track_id,reco_id");
      tables.emplace_back("ranking", "reco_id,rank");

      for (auto const& [key, header] : tables) {
        csv.emplace(key, save_to(key));
        csv[key] << header << '\n';
      }
    }
    
    // setup meta
    LARCV_INFO() << "Retrieving 3D meta..." << std::endl;
    auto meta3d = get_meta3d(mgr);

		std::unordered_set<VoxelID_t> true_voxel_ids;
		std::unordered_set<VoxelID_t> reco_voxel_ids;

    //std::ofstream log("debug_" + std::to_string(event_id) + ".csv");
    //log << "reco_id,track_id,rank,box1,box2,box3,dist12,dist13,dist23\n";
    
    // Step 1. ... create a list of true hits
    //

    // Get geometry info handler
    auto geop = lar::providerFrom<geo::Geometry>();

    // Create a hit list container
    // true_hit_vv[ch][i_hit]
    std::vector<std::vector<TrueHit_t> > true_hit_vv;
    true_hit_vv.resize(geop->Nchannels());

    // Fill a hit list container
    for(auto const& sch : LArData<supera::LArSimCh_t>()){
      // Get a unique readout channel number
      auto ch = sch.Channel();

      // Loop over hits and store
      for (auto const tick_ides : sch.TDCIDEMap()) {
        TrueHit_t hit;
        hit.time = supera::TPCTDC2Tick(tick_ides.first);

        for (auto const& edep : tick_ides.second) {
          double x_pos = (supera::TPCTDC2Tick(tick_ides.first) * supera::TPCTickPeriod() 
            - supera::TriggerOffsetTPC()) * supera::DriftVelocity();
          if(_use_true_pos) x_pos = edep.x;

          auto vox_id = meta3d.id(x_pos, edep.y, edep.z);

      	  if(vox_id == larcv::kINVALID_VOXELID) continue;
          true_voxel_ids.insert(vox_id);

          if (_dump_to_csv) 
            csv["simch"] 
              << ch << ','
              << hit.time << ','
              << edep.trackID << ','
              << edep.numElectrons << ','
              << edep.energy << ','
              << edep.x << ','
              << edep.y << ','
              << edep.z << ','
              << meta3d.id_to_x_index(vox_id) << ','
              << meta3d.id_to_y_index(vox_id) << ','
              << meta3d.id_to_z_index(vox_id) << ','
              << vox_id << '\n';

          if (edep.numElectrons < _hit_threshold_ne) continue;

          hit.voxel_ids.push_back(vox_id);
          hit.track_ids.push_back(edep.trackID);
          hit.n_electrons.push_back(edep.numElectrons);
        }

        // append hit to true_hit_vv[ch]
        if(!hit.track_ids.empty())
          true_hit_vv[ch].push_back(std::move(hit));
      }
    }

    LARCV_INFO() << "Created a list of true hits: " << true_hit_vv.size() << " channels" << std::endl;
    if(_debug) {
      size_t num_hits = 0;
      for(auto const& true_hit_v : true_hit_vv) num_hits += true_hit_v.size();
      LARCV_INFO() << " ... corresponds to " << num_hits << " true hits!" << std::endl;
    }

    // ---------------------------------------------------
    // Loop over 3d space point
    // For each 3d point, find the associated reco 2d hits
    // Match reco 2d hit -> sim hit
    // Check whether all sim hits are originated from the
    // same ionization position (in voxels)
    // ---------------------------------------------------

    auto const *ev = GetEvent();
    auto space_pts = ev->getValidHandle<std::vector<recob::SpacePoint>>(_sps_producer);

    // TODO(2020-03-20 kvtsang)  No space point, warnning?
    if (!space_pts.isValid()) return true;
    art::FindManyP<recob::Hit> hit_finder(space_pts, *ev, _sps_producer);

    std::vector<std::tuple<VoxelID_t, TrackID_t, double>> scores;

    for (size_t i = 0; i < space_pts->size(); ++i) {
      auto const &pt = space_pts->at(i);
      auto *xyz = pt.XYZ();

      auto reco_voxel_id = meta3d.id(xyz[0], xyz[1], xyz[2]);
      if(reco_voxel_id == larcv::kINVALID_VOXELID) continue;
      reco_voxel_ids.insert(reco_voxel_id);

      // reco output
      RecoVoxel3D reco_out_id(reco_voxel_id);

      std::vector<art::Ptr<recob::Hit>> hits;
      hit_finder.get(i, hits);

      std::vector<Track2Voxel> matched_segments;
      std::vector<std::set<TrackID_t>> matched_track_ids;

      if (_dump_to_csv)
        csv["cluster3d"]
          << reco_voxel_id << ','
          << xyz[0] << ','
          << xyz[1] << ','
          << xyz[2] << ','
          << pt.ErrXYZ()[1];


      for (auto const& hit_ptr : hits) {
        auto const& hit = *hit_ptr;
        auto ch = hit.Channel();
        auto t = hit.PeakTime();

        double t_start = t - _hit_window_ticks;
        double t_end = t + _hit_window_ticks;

        // match track segment between t_start and t_end
        auto segments = match_track_segments(true_hit_vv[ch], t_start, t_end);

        // save unique track_id 
        std::set<TrackID_t> track_ids;
        for (auto const& key_value : segments)
          track_ids.insert(key_value.first);

        matched_track_ids.push_back(std::move(track_ids));
        matched_segments.push_back(std::move(segments));

        if (_dump_to_csv)
          csv["cluster3d"] << ','
            << hit.Channel() << ','
            << hit.PeakTime() << ','
            << hit.RMS();
      } // for hits
      if (_dump_to_csv) csv["cluster3d"] << '\n';

      // TODO(2020-04-15 kvtang) Implement n-fold matching or n_planes != 3
      if (matched_segments.size() != 3) continue;

      // find overlapping tracks
      auto track_ids = find_overlaps(matched_track_ids);

      // build true2reco from hits
      for (auto track_id : track_ids) {
        
        RecoVoxel3D reco_out_id(reco_voxel_id);
        [[maybe_unused]] auto dist = [&](const VoxelID_t& v0, const VoxelID_t& v1)
        {
          size_t idx0[3], idx1[3];
          meta3d.id_to_xyz_index(v0, idx0[0], idx0[1], idx0[2]);
          meta3d.id_to_xyz_index(v1, idx1[0], idx1[1], idx1[2]);

          size_t d = 0;
          for (size_t i = 0; i < 3; ++i)
            d += idx0[i] > idx1[i] ? idx0[i] - idx1[i] : idx1[i] - idx0[i];

          return d;
        };

        // -----------------------------------------------------------------------
        // TODO(2020-04-15 kvtsang) Rewrite a generic mathcing for N_planes != 3
        // The current implementation assume 3 planes
        // yes, it's ugly and not robust
        // -----------------------------------------------------------------------
        for (auto v0 : matched_segments[0][track_id]) {
          for (auto v1 : matched_segments[1][track_id]) {
            if (dist(v0, v1) > 1) continue;
            for (auto v2 : matched_segments[2][track_id]) {
              if (dist(v0, v2) >  1 || dist(v1, v2) > 1) continue;
              for (VoxelID_t id : {v0, v1, v2}) {
                TrackVoxel_t true_out_id(id, track_id);
                insert_one_to_many(_true2reco, true_out_id, reco_out_id);
                insert_one_to_many(_reco2true, reco_out_id, true_out_id);
              }
            } // for v2
          } // for v1
        } // for v0
      } // loop track_ids

      if (_dump_to_csv) {
        csv["ranking"]
          << reco_voxel_id << ','
          << get_rank(reco_voxel_id, meta3d, true_voxel_ids) 
          << '\n';
      }
    } // end looping reco pts

    // -----------------------------------------------------------------------
    // TODO(2020-04-08 kvtsang) Remove this part?
    // Write out maksed_true and maske_true2reco in larcv format
    // It is kept to maintain backward compatibility.
    // Could be removed if this class is called inside SuperaMCParticleCluster
    // -----------------------------------------------------------------------
    //
    // store corresponding reco points in VoxelSetArray (outer index == true VoxelSet index)
    auto true2reco = contract_true2reco();
    auto reco2true = contract_reco2true();

    LARCV_INFO()
      << true2reco.size() << " true points mapped to " 
      << reco2true.size() << " reco points" << std::endl;

    if(!_output_tensor3d.empty() || !_output_cluster3d.empty()) {
      EventSparseTensor3D* event_tensor3d = nullptr;
      EventClusterVoxel3D* event_cluster3d = nullptr;
      if(!_output_tensor3d.empty()) {
        event_tensor3d = (larcv::EventSparseTensor3D*)(mgr.get_data("sparse3d",_output_tensor3d));
        event_tensor3d->reserve(true2reco.size());
        event_tensor3d->meta(meta3d);
      }
      if(!_output_cluster3d.empty()) {
        event_cluster3d = (larcv::EventClusterVoxel3D*)(mgr.get_data("cluster3d",_output_cluster3d));
        event_cluster3d->resize(true2reco.size());
        event_cluster3d->meta(meta3d);
      }
      
      size_t cluster_ctr=0;
      for(auto const& keyval : true2reco) {
        if(event_cluster3d && keyval.second.empty()) continue;
        if(event_tensor3d) event_tensor3d->emplace(keyval.first, 0., true);
        auto& vs = event_cluster3d->writeable_voxel_set(cluster_ctr);
        vs.reserve(keyval.second.size());
        for(auto const& reco_id: keyval.second) vs.emplace(reco_id, 0., true);
        ++cluster_ctr;
      }
    }

    if (_dump_to_csv) {
      auto gaus_hits = ev->getValidHandle<std::vector<recob::Hit>>("gaushit");
      if (gaus_hits.isValid()) {
        for (auto const& hit : *gaus_hits)
          csv["gaushits"]
            << hit.Channel() << ','
            << hit.PeakTime() << ','
            << hit.RMS() << ','
            << hit.PeakAmplitude() << ','
            << hit.Integral() << '\n';
      }

      for (auto id : true_voxel_ids)
        csv["true3d"] << id << ','
          << meta3d.pos_x(id) << ','
          << meta3d.pos_y(id) << ','
          << meta3d.pos_z(id) << '\n';

      for (auto id : reco_voxel_ids)
        csv["reco3d"] << id << ','
          << meta3d.pos_x(id) << ','
          << meta3d.pos_y(id) << ','
          << meta3d.pos_z(id) << '\n';

      for (auto id : reco_voxel_ids)
        csv["ghosts"]
          << id << ','
          << is_ghost(id) << '\n';

      for (auto& [true_id, reco_ids] : _true2reco) {
        for (auto& reco_id : reco_ids) {
          csv["true2reco"]
            << true_id.voxel_id<< ','
            << true_id.track_id << ','
            << reco_id.get_id() << '\n';
          }
      }
    }
    return true;
  }

  Track2Voxel SuperaTrue2RecoVoxel3D::match_track_segments(
      const std::vector<TrueHit_t>& hits, double t_start, double t_end)
  {
    Track2Voxel result;
    for (auto const& hit: hits) {
      // TODO(2020-03-02 kvtsang) binary search?
      // Hit times are sorted.
      // In principle could use binary search if need better performance
      if (t_start <= hit.time && hit.time <= t_end) {
        for (size_t i = 0; i < hit.track_ids.size(); ++i) {
          auto track_id = hit.track_ids[i];
          auto voxel_id = hit.voxel_ids[i];
          insert_one_to_many(result, track_id, voxel_id);
        } // loop hit.time
      } // time window
    } // loop hits
    return result;
  }

  void SuperaTrue2RecoVoxel3D::clear_maps()
  {
    _true2reco.clear();
    _reco2true.clear();
  }

  const std::unordered_set<TrackVoxel_t>&
  SuperaTrue2RecoVoxel3D::find_true(VoxelID_t reco_id) const
  {
    RecoVoxel3D reco_pt(reco_id);
    auto itr = _reco2true.find(reco_pt);
    return itr == _reco2true.end() ? _empty_true : itr->second;
  }

  const std::unordered_set<RecoVoxel3D>&
  SuperaTrue2RecoVoxel3D::find_reco(int track_id, VoxelID_t true_id) const
  {
    TrackVoxel_t true_pt(true_id, track_id);
    auto itr = _true2reco.find(true_pt);
    return itr == _true2reco.end() ? _empty_reco : itr->second;
  }

  bool SuperaTrue2RecoVoxel3D::is_ghost(VoxelID_t reco_id) const
  {
    return find_true(reco_id).size() == 0;
  }

  std::map<VoxelID_t, std::unordered_set<VoxelID_t>>
  SuperaTrue2RecoVoxel3D::contract_true2reco() 
  {
    std::map<VoxelID_t, std::unordered_set<VoxelID_t>> true2reco;
    for (auto& [true_pt,  reco_pts] : _true2reco)
      for (auto& reco_pt : reco_pts)
        insert_one_to_many(true2reco, true_pt.voxel_id, reco_pt.get_id());
    return true2reco;
  }

  std::map<VoxelID_t, std::unordered_set<VoxelID_t>>
  SuperaTrue2RecoVoxel3D::contract_reco2true()
  {
    std::map<VoxelID_t, std::unordered_set<VoxelID_t>> reco2true;
    for (auto& [reco_pt,  true_pts] : _reco2true)
      for (auto& true_pt : true_pts)
        insert_one_to_many(reco2true, reco_pt.get_id(), true_pt.voxel_id);
    return reco2true;
  }

  size_t SuperaTrue2RecoVoxel3D::get_rank(
      VoxelID_t id, const Voxel3DMeta& meta,
      const std::unordered_set<VoxelID_t>& other_ids, 
      int max_rank) {

    std::vector<std::vector<std::tuple<int, int, int>>> _rank_indices;

    for (int rank = _rank_indices.size(); rank < max_rank; ++rank) {
      _rank_indices.emplace_back();
      auto& indices = _rank_indices.back();
      for (int i = -rank; i <= rank; ++i) {
        for (int j = -rank; j <= rank; ++j) {
          for (int k = -rank; k <= rank; ++k) {
            if ((abs(i) + abs(j) + abs(k)) != rank) continue;
            indices.emplace_back(i, j, k);
          } // for k
        } // for j
      } // for  i
    } // for rank

    size_t rank = 0;
    while (rank < _rank_indices.size()) {
      for (auto [i, j, k] : _rank_indices[rank]) {
        auto id_shifted = meta.shift(id, i, j, k);
        if (other_ids.count(id_shifted) > 0)
          return rank;
      }
      ++rank;
    }
    return rank;
  }

  void SuperaTrue2RecoVoxel3D::finalize()
  {}
}

#endif
