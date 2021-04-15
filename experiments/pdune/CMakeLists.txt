set( LARCV_LIB -L$ENV{LARCV_LIBDIR} -llarcv )

set( DUNE_LIB -L$ENV{DUNE_LIB} )

include_directories ( $ENV{LARCV_INCDIR} )

link_libraries( ${LIB_NAME} -L$ENV{BOOST_LIB} -lboost_system ${ROOTLIB} ${LARCV_LIB}) # All execs need this.

art_make( BASENAME_ONLY
    LIBRARY_NAME
        dune_LArCVImageMaker

    LIB_LIBRARIES
		nurandom_RandomUtils_NuRandomService_service
		larcore_Geometry_Geometry_service
		lardataalg_DetectorInfo
		larcorealg_Geometry
		larcoreobj_SummaryData
		lardataobj_Simulation
		lardataobj_MCBase
		lardataobj_RawData
		lardataobj_RecoBase
		nusimdata_SimulationBase
		${ART_FRAMEWORK_SERVICES_REGISTRY}
        art_Persistency_Common
		art_Persistency_Provenance
		art_Utilities
		canvas
		${CETLIB}
		${CETLIB_EXCEPT}
		${ROOT_BASIC_LIB_LIST}
		${CLHEP}
		${ART_FRAMEWORK_CORE}
		${ART_FRAMEWORK_PRINCIPAL}
        ${ART_FRAMEWORK_BASIC}

    MODULE_LIBRARIES
		dune_LArCVImageMaker
		#nutools_ParticleNavigation
		nurandom_RandomUtils_NuRandomService_service
		larcore_Geometry_Geometry_service
		larcorealg_Geometry
		#larsim_Simulation
		#lardata_Utilities
		lardataalg_DetectorInfo
		larcoreobj_SummaryData
		lardataobj_Simulation
		lardataobj_MCBase
		lardataobj_RawData
		lardataobj_RecoBase
		#lardata_RecoObjects
		nusimdata_SimulationBase
		#${LARRECO_LIB}
		#${LARDATA_LIB}
		${ART_FRAMEWORK_CORE}
		${ART_FRAMEWORK_PRINCIPAL}
        ${ART_FRAMEWORK_BASIC}
		${ART_FRAMEWORK_SERVICES_REGISTRY}
		#${ART_FRAMEWORK_SERVICES_OPTIONAL}
		#${ART_FRAMEWORK_SERVICES_OPTIONAL_TFILESERVICE_SERVICE}
		art_Framework_Services_Optional_RandomNumberGenerator_service
		art_Persistency_Common
		art_Persistency_Provenance
		art_Utilities
		canvas
		${MF_MESSAGELOGGER}
		${MF_UTILITIES}
		${FHICLCPP}
		${CETLIB}
		${CETLIB_EXCEPT}
		${ROOT_BASIC_LIB_LIST}
		${CLHEP}
		)

install_headers()
install_source()
#add_subdirectory(job/pdune)
add_subdirectory(LArTPCEventGenerator)
