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
#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/math/vector3.h>
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

CVector3 ParseVector3(const std::string& str) {
    std::istringstream iss(str);
    float x, y, z;
    char comma; // To consume the commas in the string

    if (!(iss >> x >> comma >> y >> comma >> z) || comma != ',') {
        throw std::invalid_argument("Invalid vector format");
    }

    return argos::CVector3(x, y, z);
}

void Template::Init(TConfigurationNode& t_tree) {

   // Access the global configuration root node
        TConfigurationNode& tRootNode = argos::CSimulator::GetInstance().GetConfigurationRoot();

        // Access the arena node from the root node
        TConfigurationNode& tArenaNode = GetNode(tRootNode, "arena");
        // Parse attributes from the arena tag
        //std::string arenaType;
        //GetNodeAttribute(tArenaNode, "type", arenaType);

        // Example: Print the arena type
        //std::cout << "Arena Type: " << arenaType << std::endl;

        std::string sizeStr;
        GetNodeAttribute(tArenaNode, "size", sizeStr);

        std::string centerStr;
        GetNodeAttribute(tArenaNode, "center", centerStr);

        Arena arena;
        // Parse size
        arena.size = ParseVector3(sizeStr);
        
        // Parse center
        arena.center = ParseVector3(centerStr);

        // Print arena details
        LOG << "Arena Center: [" << arena.center.GetX() << ", " << arena.center.GetY() << ", " << arena.center.GetZ() << "]" << std::endl;
        LOG << "Arena Size: [" << arena.size.GetX() << ", " << arena.size.GetY() << ", " << arena.size.GetZ() << "]" << std::endl;


    // Parsing all floor circles
    TConfigurationNodeIterator it_obj("objective");
    TConfigurationNode objectiveParameters;
    try{
      // Finding all floor circle
      for ( it_obj = it_obj.begin( &t_tree ); it_obj != it_obj.end(); it_obj++ )
      {
            objectiveParameters = *it_obj;
            Objective obj;

            // Get the type attribute from the objective
            GetNodeAttribute(objectiveParameters, "type", obj.type);

            // Now, find the objective-params nested within the objective
            TConfigurationNodeIterator it_params("objective-params");
            TConfigurationNode objectiveParamsParameters;

            for (it_params = it_params.begin(&objectiveParameters); it_params != it_params.end(); it_params++) {
                objectiveParamsParameters = *it_params; // Get the current objective-params node

                // Extract attributes from objective-params
                if (obj.type == "aggregation") {
                  GetNodeAttribute(objectiveParamsParameters, "target-color", obj.target_color);
                  GetNodeAttribute(objectiveParamsParameters, "radius", obj.radius);
                } else if (obj.type == "connection") {
                  GetNodeAttribute(objectiveParamsParameters, "conn_start", obj.conn_start);
                  GetNodeAttribute(objectiveParamsParameters, "conn_end", obj.conn_end);
                  GetNodeAttribute(objectiveParamsParameters, "connection_range", obj.connection_range);
                } else if (obj.type == "foraging") {
                  GetNodeAttribute(objectiveParamsParameters, "source", obj.source);
                  GetNodeAttribute(objectiveParamsParameters, "sink", obj.sink);
                  
                }
            }

            // Store the objective
            objective = obj;
      }
      LOG << "found objective: '" << objective.type << "' color: '" << objective.target_color << "' radius: '" << objective.radius << "'"  << "' conn_start: '" << objective.conn_start << "'"    << "' connrange: '" << objective.connection_range << "'"  << std::endl;

    } catch(std::exception e) {
      LOGERR << "Problem while searching objectives" << e.what() << std::endl;
    }

    TConfigurationNodeIterator it_light("light");
    TConfigurationNode lightParameters;

    try {
        // Finding all light nodes
        for (it_light = it_light.begin(&tArenaNode); it_light != it_light.end(); it_light++) {
            lightParameters = *it_light;
            Light light;

            // Parse attributes
            std::string positionStr, orientationStr, colorStr, mediumStr, idStr;
            Real intensityValue;

            GetNodeAttribute(lightParameters, "id", idStr);
            GetNodeAttribute(lightParameters, "position", positionStr);
            GetNodeAttribute(lightParameters, "orientation", orientationStr);
            GetNodeAttribute(lightParameters, "color", colorStr);
            GetNodeAttribute(lightParameters, "intensity", intensityValue);
            GetNodeAttribute(lightParameters, "medium", mediumStr);

            // Parse position and orientation
            light.id = idStr;
            light.position = ParseVector3(positionStr);
            light.orientation = ParseVector3(orientationStr);
            light.color = colorStr;
            light.intensity = intensityValue;
            light.medium = mediumStr;

            LOG << light.id << " " << light.position << " " << light.orientation << " " << light.color << " " << light.intensity << " " << light.medium << " "<< std::endl;

            // Add the light to the list
            lLights.push_back(light);
        }

        LOG << "Number of lights: " << lLights.size() << std::endl;
    } catch (std::exception& e) {
        LOGERR << "Problem while searching for lights: " << e.what() << std::endl;
    }

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

  if (objective.type == "foraging") {
    // Retrieve source and sink colors from the objective struct
    std::string sourceColor = objective.source;
    std::string sinkColor = objective.sink;

    Circle* sourceCircle = nullptr;
    Circle* sinkCircle = nullptr;

    // Find the source and sink circles based on the colors
    for (const auto& circle : lCircles) {
        if (circle.color == sourceColor) {
            sourceCircle = const_cast<Circle*>(&circle);
        } else if (circle.color == sinkColor) {
            sinkCircle = const_cast<Circle*>(&circle);
        }
        // Break early if both circles are found
        if (sourceCircle && sinkCircle) {
            break;
        }
    }

    if (!sourceCircle || !sinkCircle) {
        LOG << "One or both circles not found. Fitness cannot be computed." << std::endl;
        m_fObjectiveFunction = 0.0f;
        return;
    }

    // Initialize counters
    UInt32 totalRobots = 0;
    UInt32 itemsPickedUp = 0;
    UInt32 itemsDropped = 0;

    CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0, 0);

    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
        CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
        
        // Get the position of the e-puck
        cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                           pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

        // Increment total robots count
        totalRobots++;

        // Check if the e-puck is inside the source circle
        Real distanceToSource = (sourceCircle->center - cEpuckPosition).Length();
        std::string strRobotId = pcEpuck->GetId();

        if (distanceToSource <= sourceCircle->radius) {
            // Pick up an item
            if (m_mapFoodData[strRobotId] == 0) { // If the robot doesn't already have food
                m_mapFoodData[strRobotId] = 1; // Mark as having food
                itemsPickedUp++;
                LOG << "Item picked up by robot: " << strRobotId << std::endl;
            }
        }

        // Check if the e-puck is inside the sink circle
        Real distanceToSink = (sinkCircle->center - cEpuckPosition).Length();
        if (distanceToSink <= sinkCircle->radius) {
            // Drop an item (count towards fitness) only if the robot has food
            if (m_mapFoodData[strRobotId] == 1) {
                itemsDropped++;
                m_mapFoodData[strRobotId] = 0; // Mark as not having food anymore
                LOG << "Item dropped by robot: " << strRobotId << std::endl;
            }
        }
    }

    // Update the fitness based on items dropped at the sink circle
    m_fObjectiveFunction += itemsDropped; // Increment fitness by the number of items dropped

    // Optionally, log the results
    std::cout << "Total items picked up: " << itemsPickedUp << std::endl;
    std::cout << "Total items dropped: " << itemsDropped << std::endl;
    std::cout << "Current Fitness: " << m_fObjectiveFunction << std::endl;
  }
  else if (objective.type == "aggregation") {
    //  -------- compute number of robots in circle by total robots ----------
    //LOG <<" computing fitness for aggregation" << std::endl;

 Circle* whiteCircle = nullptr; // actually target circle

        for (const auto& circle : lCircles) {
            if (circle.color == objective.target_color) {
                whiteCircle = const_cast<Circle*>(&circle); // If you need to modify it later
                break; // Exit the loop once we find the white circle
            }
        }

        if (whiteCircle) {
            LOG << "Found a white circle with radius: " << whiteCircle->radius << std::endl;
        } else {
            LOGERR << "No white circle found." << std::endl;
        }

// Initialize counters
UInt32 totalRobots = 0;
UInt32 robotsInWhiteCircle = 0;

CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
CVector2 cEpuckPosition(0, 0);
if (whiteCircle) {
    Real whiteCircleRadius = whiteCircle->radius; // Assuming radius is of type Real
    CVector2 whiteCircleCenter = whiteCircle->center; // Assuming center is of type CVector2

    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
        CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);
        
        // Increment total robots count
        totalRobots++;

        // Get the position of the epuck
        cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                           pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

        // Check if the epuck is inside the white circle
        Real distanceToWhiteCircle = (whiteCircleCenter - cEpuckPosition).Length();
        if (distanceToWhiteCircle <= whiteCircleRadius) {
            robotsInWhiteCircle++;
        }
    }

    // Calculate fitness as the ratio of robots in the white circle to total robots
    Real fitness = (totalRobots > 0) ? static_cast<Real>(robotsInWhiteCircle) / totalRobots : 0.0f;

    // Optionally, you can store or print the fitness value
    m_fObjectiveFunction = fitness; // just use most recent result
    std::cout << "Fitness (robots in white circle / total robots): " << m_fObjectiveFunction << std::endl;
}

  }
  else if (objective.type == "connection") {
  //LOG << "Computing fitness for connection mission" << std::endl;

    // Retrieve colors and connection range from the objective struct
    std::string connStartColor = objective.conn_start;
    std::string connEndColor = objective.conn_end;
    Real connectionRange = objective.connection_range;

    Circle* startCircle = nullptr;
    Circle* endCircle = nullptr;

    // Find the start and end circles based on the colors
    //LOGERR << "searching colors" << std::endl;
    //LOGERR << connStartColor << " " << connEndColor << std::endl;
    for (const auto& circle : lCircles) {
        if (circle.color == connStartColor) {
            startCircle = const_cast<Circle*>(&circle);
        } else if (circle.color == connEndColor) {
            endCircle = const_cast<Circle*>(&circle);
        }
        // Break early if both circles are found
        if (startCircle && endCircle) {
            break;
        }
    }

    if (!startCircle || !endCircle) {
        LOGERR << "One or both circles not found. Fitness cannot be computed." << std::endl;
        m_fObjectiveFunction = 0.0f;
        return;
    }

    // Initialize total fitness
    Real totalFitness = 0.0f;

    // Calculate the direction vector from start to end circle
    CVector2 direction = endCircle->center - startCircle->center;
    Real distanceBetweenCircles = direction.Length();
    direction.Normalize(); // Normalize the direction vector

    // Sample points along the line between the two circles
    UInt32 numSamples = static_cast<UInt32>(distanceBetweenCircles / connectionRange);
    
    CSpace::TMapPerType& tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0, 0);

    for (UInt32 i = 0; i <= numSamples; ++i) {
        // Calculate the position of the sampled point
        CVector2 sampledPoint = startCircle->center + direction * (i * connectionRange);

        // Initialize distance sum for this sampled point
        Real distanceSum = 0.0f;

        // Count how many robots are near this sampled point
        for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it) {
            CEPuckEntity* pcEpuck = any_cast<CEPuckEntity*>(it->second);

            // Get the position of the epuck
            cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                               pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

            // Calculate the distance to the sampled point
            Real distanceToSampledPoint = (sampledPoint - cEpuckPosition).Length();
            //if (distanceToSampledPoint <= connection_range) {
              distanceSum = min(-distanceToSampledPoint,distanceSum);
            //}
        }

        // Add the distance sum for this sampled point to the total fitness
        totalFitness += distanceSum;
    }

    // Store or print the fitness value
    m_fObjectiveFunction = totalFitness; // just use most recent result
    std::cout << "Total Fitness (neg sum of distances to closest robots): " << m_fObjectiveFunction << std::endl;
  }
  else if (objective.type == "distribution") {

  }
  else {
    LOGERR << "objective '"<<objective.type <<"' not implemented" << std::endl;
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
