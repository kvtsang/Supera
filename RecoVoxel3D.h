/**
 * \file RecoVoxel3D.h
 *
 * \ingroup Supera
 * 
 * \brief Class def header for a class RecoVoxel3D
 *
 * @author kvtsang
 */

/** \addtogroup Supera

    @{*/
#ifndef SUPERA_RECOVOXEL3D_H
#define SUPERA_RECOVOXEL3D_H

#include <functional>
#include <vector>
//#include "larcv/core/DataFormat/DataFormatTypes.h"
#include "larcv/core/DataFormat/Point.h"
#include "larcv/core/DataFormat/Voxel3D.h"

namespace larcv
{
  typedef long TrackID_t;

  // composite index of (voxel_id, track_id)
  struct TrackVoxel_t
  {
    VoxelID_t voxel_id;
    int track_id;

    TrackVoxel_t(VoxelID_t v, int t) : voxel_id(v), track_id(t) {}

    friend inline bool operator< (const TrackVoxel_t &lhs, const TrackVoxel_t &rhs){
      if (lhs.voxel_id == rhs.voxel_id)
        return lhs.track_id < rhs.track_id;
      return lhs.voxel_id < rhs.voxel_id;
    }

    friend inline bool operator== (const TrackVoxel_t &lhs, const TrackVoxel_t &rhs) {
      return (lhs.track_id == rhs.track_id && lhs.voxel_id == rhs.voxel_id);
    }
  };

  // minimal hit info (time, track_ids, x, y, z, Ne)
  struct TrueHit_t
  {
    double time;
    std::vector<TrackID_t> track_ids;
    std::vector<VoxelID_t> voxel_ids;
    std::vector<double> n_electrons;
  };

  /**
     \class RecoVoxel3D
     User defined class RecoVoxel3D... these comments are used to generate
     doxygen documentation!
  */
  class RecoVoxel3D
  {
    public:
      
      /// Default constructor
      RecoVoxel3D(VoxelID_t id, double charge=0) : _voxel_id(id), _charge(charge) {}
      
      /// Default(destructor
      ~RecoVoxel3D(){}

      VoxelID_t get_id() const { return _voxel_id; } 
      double get_charge() const { return _charge; } 

      void set_charge(double charge, bool is_add=false) {
        if (is_add)
          _charge += charge;
        else
          _charge = charge;
      }
      
      friend inline bool operator< (const RecoVoxel3D& lhs, const RecoVoxel3D& rhs) { 
        return lhs._voxel_id < rhs._voxel_id;
      }

      friend inline bool operator== (const RecoVoxel3D& lhs, const RecoVoxel3D& rhs) { 
        return lhs._voxel_id == rhs._voxel_id;
      }


    private:
      VoxelID_t _voxel_id; // voxel id
      double _charge;      // reco charge
  };

}

// hash functions for std::unordered_set
namespace std 
{
  template<>
  class hash<larcv::RecoVoxel3D> {
    public:
      size_t operator()(const larcv::RecoVoxel3D& pt) const {
        return pt.get_id();
      }
  };

  template<>
  class hash<larcv::TrackVoxel_t> {
    public:
      size_t operator()(const larcv::TrackVoxel_t& v) const {
        size_t seed = std::hash<int>{}(v.track_id);
        return v.voxel_id ^ (seed << 1);
      }
  };
}

namespace larcv
{
  typedef std::unordered_set<TrackVoxel_t> TrackVoxelSet;
  typedef std::unordered_set<RecoVoxel3D> RecoVoxelSet;
  typedef std::unordered_set<VoxelID_t> VoxelIDSet;
  typedef std::unordered_map<TrackVoxel_t, RecoVoxelSet> True2Reco;
  typedef std::unordered_map<RecoVoxel3D, TrackVoxelSet> Reco2True;
  //typedef std::unordered_map<TrackID_t, VoxelIDSet> Track2Voxel;
  typedef std::map<TrackID_t, VoxelIDSet> Track2Voxel;
}
#endif
