/**
  * @file <loop-functions/example/ForagingLoopFunc.h>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @package ARGoS3-AutoMoDe
  *
  * @license MIT License
  */

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