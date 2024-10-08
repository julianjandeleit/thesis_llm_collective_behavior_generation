/**
  * @file <loop-functions/ForagingTwoSpotsLoopFunc.cpp>
  *
  * @author Antoine Ligot - <aligot@ulb.ac.be>
  *
  * @license MIT License
  */

#include "Template.h"
#include <iostream>
#include <fstream>
using namespace std;

/****************************************/
/****************************************/

Template::Template() {
  m_cCoordSpot1 = CVector2(0.75,0);
  m_fObjectiveFunction = 0;
}


/****************************************/
/****************************************/

Template::Template(const Template& orig) {
}

/****************************************/
/****************************************/

void Template::Init(TConfigurationNode& t_tree) {
    // Parsing all floor circles
    TConfigurationNodeIterator it_circle("circle");
    TConfigurationNode circleParameters;
    try{
      // Finding all floor circle
      for ( it_circle = it_circle.begin( &t_tree ); it_circle != it_circle.end(); it_circle++ )
      {
          circleParameters = *it_circle;
          Circle c;
          GetNodeAttribute(circleParameters, "position", c.center);
          GetNodeAttribute(circleParameters, "radius", c.radius);
          GetNodeAttribute(circleParameters, "color", c.color);
          lCircles.push_back(c);
      }
      LOG << "number of floor circle: " << lCircles.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching floor circles" << std::endl;
    }

    // Parsing all floor rectangles
    TConfigurationNodeIterator it_rect("rectangle");
    TConfigurationNode rectParameters;
    try{
      // Finding all floor circle
      for ( it_rect = it_rect.begin( &t_tree ); it_rect != it_rect.end(); it_rect++ )
      {
          rectParameters = *it_rect;
          Rectangle r;
          GetNodeAttribute(rectParameters, "center", r.center);
          GetNodeAttribute(rectParameters, "angle", r.angle);
          GetNodeAttribute(rectParameters, "width", r.width);
          GetNodeAttribute(rectParameters, "height", r.height);
          GetNodeAttribute(rectParameters, "color", r.color);
          lRectangles.push_back(r);
      }
      LOG << "number of floor rectangles: " << lRectangles.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching floor circles" << std::endl;
    }

    // Parsing all spawning circles areas
    TConfigurationNodeIterator it_initCircle("spawnCircle");
    TConfigurationNode initCircleParameters;
    try{
      // Finding all floor circle
      for ( it_initCircle = it_initCircle.begin( &t_tree ); it_initCircle != it_initCircle.end(); it_initCircle++ )
      {
          initCircleParameters = *it_initCircle;
          Circle c;
          GetNodeAttribute(initCircleParameters, "position", c.center);
          GetNodeAttribute(initCircleParameters, "radius", c.radius);
          initCircles.push_back(c);
      }
      LOG << "number of spawning circles areas: " << initCircles.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching floor circles" << std::endl;
    }

    // Parsing all spawning rectangles areas
    TConfigurationNodeIterator it_initRect("spawnRectangle");
    TConfigurationNode initRectParameters;
    try{
      // Finding all floor circle
      for ( it_initRect = it_initRect.begin( &t_tree ); it_initRect != it_initRect.end(); it_initRect++ )
      {
          initRectParameters = *it_initRect;
          Rectangle r;
          GetNodeAttribute(initRectParameters, "center", r.center);
          GetNodeAttribute(initRectParameters, "angle", r.angle);
          GetNodeAttribute(initRectParameters, "width", r.width);
          GetNodeAttribute(initRectParameters, "height", r.height);
          initRectangles.push_back(r);
      }
      LOG << "number of spawning rectangles areas: " << initRectangles.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching floor circles" << std::endl;
    }

    // Parsing all epucks manualy set positions
    TConfigurationNodeIterator it_epuck("epuck");
    TConfigurationNode epuckParameters;
    try{
      // Finding all epucks positions
      for ( it_epuck = it_epuck.begin( &t_tree ); it_epuck != it_epuck.end(); it_epuck++ )
      {
          epuckParameters = *it_epuck;
          CVector2 e;
          GetNodeAttribute(epuckParameters, "position", e);
          epucks.push_back(e);
      }
      LOG << "number of manualy placed epucks: " << epucks.size() << std::endl;
    } catch(std::exception e) {
      LOGERR << "Problem while searching manual epuck positions" << std::endl;
    }

    CoreLoopFunctions::Init(t_tree);
}

/****************************************/
/****************************************/

Template::~Template() {
}

/****************************************/
/****************************************/

void Template::Destroy() {}

/****************************************/
/****************************************/

argos::CColor Template::GetFloorColor(const argos::CVector2& c_position_on_plane) {
  CVector2 vCurrentPoint(c_position_on_plane.GetX(), c_position_on_plane.GetY());
  if (IsOnColor(vCurrentPoint, "black")) {
    return CColor::BLACK;
  }

  if (IsOnColor(vCurrentPoint, "white")) {
    return CColor::WHITE;
  }

  else {
    return CColor::GRAY50;
  }
}

/****************************************/
/****************************************/

bool Template::IsOnColor(CVector2& c_position_on_plane, std::string color) {
  // checking floor circles
  for (Circle c : lCircles) 
  {
    if (c.color == color)
    {
      Real d = (c.center - c_position_on_plane).Length();
      if (d <= c.radius) 
      {
        return true;
      }
    }
  }

  // checking floor rectangles
  for (Rectangle r : lRectangles) 
  {
    if (r.color == color)
    {
      Real phi = std::atan(r.height/r.width);
      Real theta = r.angle * (M_PI/180);
      Real hyp = std::sqrt((r.width*r.width) + (r.height*r.height));
      // compute position of three corner of the rectangle
      CVector2 corner1 = CVector2(r.center.GetX() - hyp*std::cos(phi + theta), r.center.GetY() + hyp*std::sin(phi + theta));
      CVector2 corner2 = CVector2(r.center.GetX() + hyp*std::cos(phi - theta), r.center.GetY() + hyp*std::sin(phi - theta));
      CVector2 corner3 = CVector2(r.center.GetX() + hyp*std::cos(phi + theta), r.center.GetY() - hyp*std::sin(phi + theta));
      // computing the three vectors
      CVector2 corner2ToCorner1 = corner1 - corner2; 
      CVector2 corner2ToCorner3 = corner3 - corner2; 
      CVector2 corner2ToPos = c_position_on_plane - corner2; 
      // compute the four inner products
      Real ip1 = corner2ToPos.GetX()*corner2ToCorner1.GetX() + corner2ToPos.GetY()*corner2ToCorner1.GetY();
      Real ip2 = corner2ToCorner1.GetX()*corner2ToCorner1.GetX() + corner2ToCorner1.GetY()*corner2ToCorner1.GetY();
      Real ip3 = corner2ToPos.GetX()*corner2ToCorner3.GetX() + corner2ToPos.GetY()*corner2ToCorner3.GetY();
      Real ip4 = corner2ToCorner3.GetX()*corner2ToCorner3.GetX() + corner2ToCorner3.GetY()*corner2ToCorner3.GetY();
      if (ip1 > 0 && ip1 < ip2 && ip3 > 0 && ip3 < ip4)
      {
        return true;
      }
    }
  }
  return false;
}

/****************************************/
/****************************************/

void Template::Reset() {
  CoreLoopFunctions::Reset();
  std::ios::sync_with_stdio(false);
  m_mapFoodData.clear();
  m_fObjectiveFunction = 0;
}

/****************************************/
/****************************************/

void Template::PostStep() {
  UInt32 score_temp = m_fObjectiveFunction;

  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);
  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);

    std::string strRobotId = pcEpuck->GetId();
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                       pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());


    Real fDistanceSpot1 = (m_cCoordSpot1 - cEpuckPosition).Length();
    Real fDistanceSpot2 = (m_cCoordSpot2 - cEpuckPosition).Length();
    if (fDistanceSpot1 <= m_fRadius) {
      m_mapFoodData[strRobotId] = 1;
    } else if (fDistanceSpot2 <= m_fRadius){
      m_mapFoodData[strRobotId] = 1;
    } else if (cEpuckPosition.GetY() <= m_fNestLimit) {
      std::map<std::string, UInt32>::iterator itFood = m_mapFoodData.find(strRobotId);
      if (itFood != m_mapFoodData.end()) {
        m_fObjectiveFunction += itFood->second;
      }
      m_mapFoodData[strRobotId] = 0;
      // LOG << "Obj " << m_fObjectiveFunction << std::endl;
    }
  }
  if (score_temp != m_fObjectiveFunction) {
     //LOGERR << "Obj " << m_fObjectiveFunction << std::endl;
  }
}

/****************************************/
/****************************************/

void Template::PostExperiment() {
  // Create and open a text file 
  ofstream MyFile("pos.mu",ios::trunc);

  CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0,0);

  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
    CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
    std::string strRobotId = pcEpuck->GetId();
    cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                        pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

    // Write to the file
    MyFile << cEpuckPosition << std::endl;
    LOG << cEpuckPosition << endl;
  }

  // Close the file
  MyFile.close();  
}

/****************************************/
/****************************************/

Real Template::GetObjectiveFunction() {
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

CVector3 Template::GetRandomPosition() {
  
  int lenQuad = initRectangles.size();
  int lenDisk = initCircles.size();
  int randArea = m_pcRng->Uniform(CRange<int>(1, lenDisk+lenQuad));
  int ind = 0;

  LOG << "epucks left before " << epucks.size() << endl;
  if(epucks.size() > 0){
    ind = rand() % epucks.size();

    std::list<CVector2>::iterator it = epucks.begin();
    advance(it, ind);
    CVector2 pos = *it;

    Real posX = pos.GetX();
    Real posY = pos.GetY();
    
    epucks.erase(it);
    LOG << "epucks after " << epucks.size() << endl;

    return CVector3(posX, posY, 0);
  }

  if (randArea > lenDisk)
  {
    int area = randArea - lenDisk;
    std::list<Rectangle>::iterator it = initRectangles.begin();
    advance(it, area-1);
    Rectangle rectArea = *it;

    Real a = m_pcRng->Uniform(CRange<Real>(-1.0f, 1.0f));
    Real b = m_pcRng->Uniform(CRange<Real>(-1.0f, 1.0f));

    Real theta = -rectArea.angle * (M_PI/180);

    Real posX = rectArea.center.GetX() + a * rectArea.width * cos(theta) - b * rectArea.height * sin(theta);
    Real posY = rectArea.center.GetY() + a * rectArea.width * sin(theta) + b * rectArea.height * cos(theta);

    return CVector3(posX, posY, 0);
  }
  else
  {
    int area = randArea;
    std::list<Circle>::iterator it = initCircles.begin();
    advance(it, area-1);
    Circle diskArea = *it;

    Real temp;
    Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
    Real b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
    // If b < a, swap them
    if (b < a) {
      temp = a;
      a = b;
      b = temp;
    }
    Real posX = diskArea.center.GetX() + b * diskArea.radius * cos(2 * CRadians::PI.GetValue() * (a/b));
    Real posY = diskArea.center.GetY() + b * diskArea.radius * sin(2 * CRadians::PI.GetValue() * (a/b));

    return CVector3(posX, posY, 0);
  }
}

REGISTER_LOOP_FUNCTIONS(Template, "template");
