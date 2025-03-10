/**
  * @file <loop-functions/example/ForagingLoopFunc.h>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

 /* adapted by Julian Jandeleit for custom fitness functions */

#ifndef TEMPLATE
#define TEMPLATE

#include <map>
#include <list>
#include <math.h> 

#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/e-puck/simulator/epuck_entity.h>

#include "../../src/CoreLoopFunctions.h"

using namespace argos;

class Template: public CoreLoopFunctions {
  public:
    Template();
    Template(const Template& orig);
    virtual ~Template();

    virtual void Destroy();
    virtual void Init(TConfigurationNode& t_tree);

    virtual argos::CColor GetFloorColor(const argos::CVector2& c_position_on_plane);
    virtual void PostStep();
    virtual void PostExperiment();
    virtual void Reset();

    Real GetObjectiveFunction();

    CVector3 GetRandomPosition();

    bool IsOnColor(CVector2& c_position_on_plane, std::string color);

  private:
    Real m_fRadius;
    Real m_fNestLimit;
    CVector2 m_cCoordSpot1;
    CVector2 m_cCoordSpot2;
    Real m_fObjectiveFunction;

    std::map<std::string, UInt32> m_mapFoodData;

    struct Circle {
      CVector2 center;
      Real radius;
      std::string color;
    };

    // ----- CUSTOM DATASTRUCTURES ------
    struct Objective {
      std::string type;

      // aggregation
      std::string target_color;
      Real radius;

      // connection
      std::string conn_start;
      std::string conn_end;
      Real connection_range;

      // distribution (dispresion)
      std::string area; // x,y as string
      // also reuses connection_range

      // foraging
      std::string source;
      std::string sink;
    };

    struct Arena {
        CVector3 center; // Center of the arena
        CVector3 size;   // Size of the arena
    };

    struct Light {
        std::string id;
        CVector3 position; // Position of the light
        CVector3 orientation; // Orientation of the light
        std::string color; // Color of the light
        Real intensity;    // Intensity of the light
        std::string medium; // Medium of the light
    };


    std::list<Light> lLights; // List to store lights
    Objective objective;

    // ----- END CUSTOM DATASTRUCTURES ------

    struct Rectangle {
      CVector2 center;
      Real width;
      Real height;
      Real angle;
      std::string color;
    };

    std::list<Circle> lCircles;
    std::list<Rectangle> lRectangles;

    std::list<Circle> initCircles;
    std::list<Rectangle> initRectangles;

    std::list<CVector2> epucks;
};

#endif