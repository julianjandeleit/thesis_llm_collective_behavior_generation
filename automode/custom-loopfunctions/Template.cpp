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

Template::Template()
{
  m_cCoordSpot1 = CVector2(0.75, 0);
  m_fObjectiveFunction = 0;
}

/****************************************/
/****************************************/

Template::Template(const Template &orig)
{
}

/****************************************/
/****************************************/

CVector3 ParseVector3(const std::string &str)
{
  std::istringstream iss(str);
  float x, y, z;
  char comma; // To consume the commas in the string

  if (!(iss >> x >> comma >> y >> comma >> z) || comma != ',')
  {
    throw std::invalid_argument("Invalid vector format");
  }

  return argos::CVector3(x, y, z);
}

void Template::Init(TConfigurationNode &t_tree)
{

  // Access the global configuration root node
  TConfigurationNode &tRootNode = argos::CSimulator::GetInstance().GetConfigurationRoot();

  // Access the arena node from the root node
  TConfigurationNode &tArenaNode = GetNode(tRootNode, "arena");
  // Parse attributes from the arena tag
  // std::string arenaType;
  // GetNodeAttribute(tArenaNode, "type", arenaType);

  // Example: Print the arena type
  // std::cout << "Arena Type: " << arenaType << std::endl;

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
  try
  {
    // Finding all floor circle
    for (it_obj = it_obj.begin(&t_tree); it_obj != it_obj.end(); it_obj++)
    {
      objectiveParameters = *it_obj;
      Objective obj;

      // Get the type attribute from the objective
      GetNodeAttribute(objectiveParameters, "type", obj.type);

      // Now, find the objective-params nested within the objective
      TConfigurationNodeIterator it_params("objective-params");
      TConfigurationNode objectiveParamsParameters;

      for (it_params = it_params.begin(&objectiveParameters); it_params != it_params.end(); it_params++)
      {
        objectiveParamsParameters = *it_params; // Get the current objective-params node

        // Extract attributes from objective-params
        if (obj.type == "aggregation")
        {
          GetNodeAttribute(objectiveParamsParameters, "target-color", obj.target_color);
          GetNodeAttribute(objectiveParamsParameters, "radius", obj.radius);
        }
        else if (obj.type == "connection")
        {
          GetNodeAttribute(objectiveParamsParameters, "conn_start", obj.conn_start);
          GetNodeAttribute(objectiveParamsParameters, "conn_end", obj.conn_end);
          GetNodeAttribute(objectiveParamsParameters, "connection_range", obj.connection_range);
        }
        else if (obj.type == "foraging")
        {
          GetNodeAttribute(objectiveParamsParameters, "source", obj.source);
          GetNodeAttribute(objectiveParamsParameters, "sink", obj.sink);
        }
        else if (obj.type == "distribution")
        { // also known as coverage
          GetNodeAttribute(objectiveParamsParameters, "area", obj.area);
          GetNodeAttribute(objectiveParamsParameters, "connection_range", obj.connection_range);
        }
      }

      // Store the objective
      objective = obj;
    }
    LOG << "found objective: '" << objective.type << "' color: '" << objective.target_color << "' radius: '" << objective.radius << "'" << "' conn_start: '" << objective.conn_start << "'" << "' connrange: '" << objective.connection_range << "'" << std::endl;
  }
  catch (std::exception e)
  {
    LOGERR << "Problem while searching objectives" << e.what() << std::endl;
  }

  TConfigurationNodeIterator it_light("light");
  TConfigurationNode lightParameters;

  try
  {
    // Finding all light nodes
    for (it_light = it_light.begin(&tArenaNode); it_light != it_light.end(); it_light++)
    {
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

      LOG << light.id << " " << light.position << " " << light.orientation << " " << light.color << " " << light.intensity << " " << light.medium << " " << std::endl;

      // Add the light to the list
      lLights.push_back(light);
    }

    LOG << "Number of lights: " << lLights.size() << std::endl;
  }
  catch (std::exception &e)
  {
    LOGERR << "Problem while searching for lights: " << e.what() << std::endl;
  }

  // Parsing all floor circles
  TConfigurationNodeIterator it_circle("circle");
  TConfigurationNode circleParameters;
  try
  {
    // Finding all floor circle
    for (it_circle = it_circle.begin(&t_tree); it_circle != it_circle.end(); it_circle++)
    {
      circleParameters = *it_circle;
      Circle c;
      GetNodeAttribute(circleParameters, "position", c.center);
      GetNodeAttribute(circleParameters, "radius", c.radius);
      GetNodeAttribute(circleParameters, "color", c.color);
      lCircles.push_back(c);
    }
    LOG << "number of floor circle: " << lCircles.size() << std::endl;
  }
  catch (std::exception e)
  {
    LOGERR << "Problem while searching floor circles" << std::endl;
  }

  // Parsing all floor rectangles
  TConfigurationNodeIterator it_rect("rectangle");
  TConfigurationNode rectParameters;
  try
  {
    // Finding all floor circle
    for (it_rect = it_rect.begin(&t_tree); it_rect != it_rect.end(); it_rect++)
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
  }
  catch (std::exception e)
  {
    LOGERR << "Problem while searching floor circles" << std::endl;
  }

  // Parsing all spawning circles areas
  TConfigurationNodeIterator it_initCircle("spawnCircle");
  TConfigurationNode initCircleParameters;
  try
  {
    // Finding all floor circle
    for (it_initCircle = it_initCircle.begin(&t_tree); it_initCircle != it_initCircle.end(); it_initCircle++)
    {
      initCircleParameters = *it_initCircle;
      Circle c;
      GetNodeAttribute(initCircleParameters, "position", c.center);
      GetNodeAttribute(initCircleParameters, "radius", c.radius);
      initCircles.push_back(c);
    }
    LOG << "number of spawning circles areas: " << initCircles.size() << std::endl;
  }
  catch (std::exception e)
  {
    LOGERR << "Problem while searching floor circles" << std::endl;
  }

  // Parsing all spawning rectangles areas
  TConfigurationNodeIterator it_initRect("spawnRectangle");
  TConfigurationNode initRectParameters;
  try
  {
    // Finding all floor circle
    for (it_initRect = it_initRect.begin(&t_tree); it_initRect != it_initRect.end(); it_initRect++)
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
  }
  catch (std::exception e)
  {
    LOGERR << "Problem while searching floor circles" << std::endl;
  }

  // Parsing all epucks manualy set positions
  TConfigurationNodeIterator it_epuck("epuck");
  TConfigurationNode epuckParameters;
  try
  {
    // Finding all epucks positions
    for (it_epuck = it_epuck.begin(&t_tree); it_epuck != it_epuck.end(); it_epuck++)
    {
      epuckParameters = *it_epuck;
      CVector2 e;
      GetNodeAttribute(epuckParameters, "position", e);
      epucks.push_back(e);
    }
    LOG << "number of manualy placed epucks: " << epucks.size() << std::endl;
  }
  catch (std::exception e)
  {
    LOGERR << "Problem while searching manual epuck positions" << std::endl;
  }

  CoreLoopFunctions::Init(t_tree);
}

/****************************************/
/****************************************/

Template::~Template()
{
}

/****************************************/
/****************************************/

void Template::Destroy() {}

/****************************************/
/****************************************/

argos::CColor Template::GetFloorColor(const argos::CVector2 &c_position_on_plane)
{
  CVector2 vCurrentPoint(c_position_on_plane.GetX(), c_position_on_plane.GetY());
  if (IsOnColor(vCurrentPoint, "black"))
  {
    return CColor::BLACK;
  }

  if (IsOnColor(vCurrentPoint, "white"))
  {
    return CColor::WHITE;
  }

  else
  {
    return CColor::GRAY50;
  }
}

/****************************************/
/****************************************/

bool Template::IsOnColor(CVector2 &c_position_on_plane, std::string color)
{
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
      Real phi = std::atan(r.height / r.width);
      Real theta = r.angle * (M_PI / 180);
      Real hyp = std::sqrt((r.width * r.width) + (r.height * r.height));
      // compute position of three corner of the rectangle
      CVector2 corner1 = CVector2(r.center.GetX() - hyp * std::cos(phi + theta), r.center.GetY() + hyp * std::sin(phi + theta));
      CVector2 corner2 = CVector2(r.center.GetX() + hyp * std::cos(phi - theta), r.center.GetY() + hyp * std::sin(phi - theta));
      CVector2 corner3 = CVector2(r.center.GetX() + hyp * std::cos(phi + theta), r.center.GetY() - hyp * std::sin(phi + theta));
      // computing the three vectors
      CVector2 corner2ToCorner1 = corner1 - corner2;
      CVector2 corner2ToCorner3 = corner3 - corner2;
      CVector2 corner2ToPos = c_position_on_plane - corner2;
      // compute the four inner products
      Real ip1 = corner2ToPos.GetX() * corner2ToCorner1.GetX() + corner2ToPos.GetY() * corner2ToCorner1.GetY();
      Real ip2 = corner2ToCorner1.GetX() * corner2ToCorner1.GetX() + corner2ToCorner1.GetY() * corner2ToCorner1.GetY();
      Real ip3 = corner2ToPos.GetX() * corner2ToCorner3.GetX() + corner2ToPos.GetY() * corner2ToCorner3.GetY();
      Real ip4 = corner2ToCorner3.GetX() * corner2ToCorner3.GetX() + corner2ToCorner3.GetY() * corner2ToCorner3.GetY();
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

void Template::Reset()
{
  CoreLoopFunctions::Reset();
  std::ios::sync_with_stdio(false);
  m_mapFoodData.clear();
  m_fObjectiveFunction = 0;
}

/****************************************/
/****************************************/

void Template::PostStep()
{

  if (objective.type == "foraging")
  {
    // Retrieve source and sink colors from the objective struct
    std::string sourceColor = objective.source;
    std::string sinkColor = objective.sink;

    Circle *sourceCircle = nullptr;
    Circle *sinkCircle = nullptr;

    // Find the source and sink circles based on the colors
    for (const auto &circle : lCircles)
    {
      if (circle.color == sourceColor)
      {
        sourceCircle = const_cast<Circle *>(&circle);
      }
      else if (circle.color == sinkColor)
      {
        sinkCircle = const_cast<Circle *>(&circle);
      }
      // Break early if both circles are found
      if (sourceCircle && sinkCircle)
      {
        break;
      }
    }

    if (!sourceCircle || !sinkCircle)
    {
      LOG << "One or both circles not found. Fitness cannot be computed." << std::endl;
      m_fObjectiveFunction = 0.0f;
      return;
    }

    // Initialize counters
    UInt32 totalRobots = 0;
    UInt32 itemsPickedUp = 0;
    UInt32 itemsDropped = 0;
    UInt32 robotOnSource = 0;
    UInt32 robotOnTarget = 0;

    CSpace::TMapPerType &tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0, 0);

    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
    {
      CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);

      // Get the position of the e-puck
      cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                         pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

      // Increment total robots count
      totalRobots++;

      // Check if the e-puck is inside the source circle
      Real distanceToSource = (sourceCircle->center - cEpuckPosition).Length();
      std::string strRobotId = pcEpuck->GetId();

      if (distanceToSource <= sourceCircle->radius)
      {
        // LOG << "distance to source" << distanceToSource << std::endl;
        robotOnSource++;
        // Pick up an item
        if (m_mapFoodData[strRobotId] == 0)
        {                                // If the robot doesn't already have food
          m_mapFoodData[strRobotId] = 1; // Mark as having food
          itemsPickedUp++;
          LOG << "Item picked up by robot: " << strRobotId << std::endl;
        }
      }

      // Check if the e-puck is inside the sink circle
      Real distanceToSink = (sinkCircle->center - cEpuckPosition).Length();
      if (distanceToSink <= sinkCircle->radius)
      {
        robotOnTarget++;
        // Drop an item (count towards fitness) only if the robot has food
        if (m_mapFoodData[strRobotId] == 1)
        {
          itemsDropped++;
          m_mapFoodData[strRobotId] = 0; // Mark as not having food anymore
          LOG << "Item dropped by robot: " << strRobotId << std::endl;
        }
      }
    }

    // Update the fitness based on items dropped at the sink circle
    m_fObjectiveFunction += itemsDropped + itemsPickedUp / 100.0f + robotOnSource / 10000.0f + robotOnTarget / 10000.0f; // Increment fitness by the number of items dropped

    // Optionally, log the results
    std::cout << "Total items picked up: " << itemsPickedUp << std::endl;
    std::cout << "Total items dropped: " << itemsDropped << std::endl;
    std::cout << "Current Fitness: " << m_fObjectiveFunction << std::endl;
  }
  else if (objective.type == "aggregation")
  {
    //  -------- compute number of robots in circle by total robots ----------
    // LOG <<" computing fitness for aggregation" << std::endl;

    Circle *whiteCircle = nullptr; // actually target circle
    Circle *otherCircle = nullptr;

    for (const auto &circle : lCircles)
    {
      if (circle.color == objective.target_color)
      {
        whiteCircle = const_cast<Circle *>(&circle); // If you need to modify it later
                                                     // break; // Exit the loop once we find the white circle
      }
      else
      {
        otherCircle = const_cast<Circle *>(&circle); // If you need to modify it later
      }
    }

    if (whiteCircle)
    {
      LOG << "Found a white circle with radius: " << whiteCircle->radius << std::endl;
    }
    else
    {
      LOGERR << "No white circle found." << std::endl;
    }

    if (whiteCircle)
    {
      LOG << "Found an other circle with radius: " << whiteCircle->radius << std::endl;
    }
    else
    {
      LOGERR << "No other circle found." << std::endl;
    }

    // Initialize counters
    UInt32 totalRobots = 0;
    UInt32 robotsInWhiteCircle = 0;

    UInt32 robotsInOtherCircle = 0;

    CSpace::TMapPerType &tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0, 0);
    if (whiteCircle)
    {
      Real whiteCircleRadius = whiteCircle->radius;     // Assuming radius is of type Real
      CVector2 whiteCircleCenter = whiteCircle->center; // Assuming center is of type CVector2

      std::vector<CVector2> robotPositions;

      for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
      {
        CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);

        // Increment total robots count
        totalRobots++;

        // Get the position of the epuck
        cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                           pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

        robotPositions.push_back(cEpuckPosition);

        // Check if the epuck is inside the white circle
        Real distanceToWhiteCircle = (whiteCircleCenter - cEpuckPosition).Length();
        if (distanceToWhiteCircle <= whiteCircleRadius)
        {
          robotsInWhiteCircle++;
        }
      }

      // Now calculate the area spanned by the robots
      Real areaSpannedByRobots = 0.0;
      int n = robotPositions.size();

      // Calculate the area using the shoelace formula
      for (int i = 0; i < n; ++i)
      {
        areaSpannedByRobots += (robotPositions[i].GetX() * robotPositions[(i + 1) % n].GetY()) -
                               (robotPositions[(i + 1) % n].GetX() * robotPositions[i].GetY());
      }

      areaSpannedByRobots = std::abs(areaSpannedByRobots) / 2.0;

      if (otherCircle)
      {
        Real otherCircleRadius = otherCircle->radius;     // Assuming radius is of type Real
        CVector2 otherCircleCenter = otherCircle->center; // Assuming center is of type CVector2

        for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
        {
          CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);

          // Increment total robots count
          totalRobots++;

          // Get the position of the epuck
          cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                             pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

          // Check if the epuck is inside the white circle
          Real distanceToCircle = (otherCircleCenter - cEpuckPosition).Length();
          if (distanceToCircle <= otherCircleRadius)
          {
            robotsInOtherCircle++;
          }
        }

        Real robotsoutsidecircles = totalRobots - robotsInWhiteCircle - robotsInOtherCircle;
        // Calculate fitness as the ratio of robots in the white circle to total robots
        Real fitness = (totalRobots > 0) ? static_cast<Real>(robotsInWhiteCircle) / totalRobots : 0.0f;
        fitness += -robotsInOtherCircle * 10 - robotsoutsidecircles; // we really want to punish stopping at wrong circle as this seems to be ignored at some optimizations
        fitness -= areaSpannedByRobots;
        // Optionally, you can store or print the fitness value
        m_fObjectiveFunction += fitness; // just use most recent result
        std::cout << "Fitness (robots in white circle / total robots): " << m_fObjectiveFunction << std::endl;
      }
    }
  }
  else if (objective.type == "connection")
  {
  }
  else if (objective.type == "distribution")
  {
    LOG << "Computing fitness for distribution mission" << std::endl;

    // Retrieve area and connection range from the objective struct
    std::string areaStr = objective.area;
    Real connectionRange = objective.connection_range;

    // Parse the area string to get width and height
    std::istringstream areaStream(areaStr);
    std::string widthStr, heightStr;
    std::getline(areaStream, widthStr, ',');
    std::getline(areaStream, heightStr, ',');
    Real areaWidth = std::stof(widthStr);
    Real areaHeight = std::stof(heightStr);
    Real targetArea = areaWidth * areaHeight;

    // Initialize bounding box coordinates
    Real minX = std::numeric_limits<Real>::max();
    Real minY = std::numeric_limits<Real>::max();
    Real maxX = std::numeric_limits<Real>::lowest();
    Real maxY = std::numeric_limits<Real>::lowest();

    CSpace::TMapPerType &tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    CVector2 cEpuckPosition(0, 0);

    // Compute the bounding box of all robots
    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
    {
      CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);

      // Get the position of the e-puck
      cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                         pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

      // Update bounding box coordinates
      minX = std::min(minX, cEpuckPosition.GetX());
      minY = std::min(minY, cEpuckPosition.GetY());
      maxX = std::max(maxX, cEpuckPosition.GetX());
      maxY = std::max(maxY, cEpuckPosition.GetY());
    }

    // Calculate the area of the bounding box
    Real boundingBoxArea = (maxX - minX) * (maxY - minY);
    Real areaDifference = abs(targetArea - boundingBoxArea);

    // Step 1.1: Calculate the center of the bounding box
    Real centerX = (minX + maxX) / 2.0;
    Real centerY = (minY + maxY) / 2.0;

    // Step 1.2: Calculate the dimensions of the area based on targetArea
    Real targetWidth = std::sqrt(targetArea); // Assuming a square area for simplicity
    Real targetHeight = targetWidth;          // For a square area

    // Step 1.3: Generate sample points centered at the bounding box center
    std::vector<CVector2> samplePoints;
    Real startX = centerX - (targetWidth / 2.0);
    Real startY = centerY - (targetHeight / 2.0);

    for (Real x = startX; x <= startX + targetWidth; x += connectionRange)
    {
      for (Real y = startY; y <= startY + targetHeight; y += connectionRange)
      {
        samplePoints.emplace_back(x, y);
      }
    }

    // Step 2: Initialize data structures
    std::map<size_t, std::vector<CEPuckEntity *>> closestRobotsMap; // Change key type to size_t
    Real totalClosestDistance = 0.0f;

    // Step 3: Enumerate robots
    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
    {
      CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);
      cEpuckPosition.Set(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                         pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

      CVector2 robotPosition = cEpuckPosition;
      Real closestDistance = std::numeric_limits<Real>::max();
      size_t closestSampleIndex = 0; // Store the index of the closest sample point

      // Step 4: Find closest sample point
      for (size_t i = 0; i < samplePoints.size(); ++i)
      { // Use index i
        const CVector2 &samplePoint = samplePoints[i];
        Real distance = (robotPosition - samplePoint).Length();
        if (distance < closestDistance)
        {
          closestDistance = distance;
          closestSampleIndex = i; // Update the index of the closest sample point
        }
      }

      // Step 5: Store closest robot information using the index
      closestRobotsMap[closestSampleIndex].push_back(pcEpuck);
      totalClosestDistance += closestDistance;
    }

    // Step 6: Count robots with multiple neighbors
    int countOfRobotsWithMultipleNeighbors = 0;
    for (const auto &entry : closestRobotsMap)
    {
      if (entry.second.size() >= 2)
      {
        countOfRobotsWithMultipleNeighbors += entry.second.size();
      }
    }

    // Step 7: Output results
    std::cout << "Total Closest Distance: " << totalClosestDistance << std::endl;
    std::cout << "Number of Robots with Multiple Neighbors: " << countOfRobotsWithMultipleNeighbors << std::endl;

    // Calculate fitness as the negative area difference and average distance
    // m_fObjectiveFunction = - areaDifference - averageDistance;
    m_fObjectiveFunction += -countOfRobotsWithMultipleNeighbors - totalClosestDistance;

    // Log the results
    // std::cout << "Bounding Box Area: " << boundingBoxArea << std::endl;
    // std::cout << "Target Area: " << targetArea << std::endl;
    // std::cout << "Area Difference: " << areaDifference << std::endl;
    // std::cout << "Average Distance to Closest Robot: " << averageDistance << std::endl;
    std::cout << "Current Fitness: " << m_fObjectiveFunction << std::endl;
  }
  else
  {
    LOGERR << "objective '" << objective.type << "' not implemented" << std::endl;
  }
}

/****************************************/
/****************************************/

void Template::PostExperiment()
{
  // fitness function computation
  if (objective.type == "distribution")
  {
  }
  else if (objective.type == "connection")
  {
    // LOG << "Computing fitness for connection mission" << std::endl;

    // Retrieve colors and connection range from the objective struct
    std::string connStartColor = objective.conn_start;
    std::string connEndColor = objective.conn_end;
    Real connectionRange = objective.connection_range;

    Circle *startCircle = nullptr;
    Circle *endCircle = nullptr;

    // Find the start and end circles based on the colors
    // LOGERR << "searching colors" << std::endl;
    // LOGERR << connStartColor << " " << connEndColor << std::endl;
    for (const auto &circle : lCircles)
    {
      if (circle.color == connStartColor)
      {
        startCircle = const_cast<Circle *>(&circle);
      }
      else if (circle.color == connEndColor)
      {
        endCircle = const_cast<Circle *>(&circle);
      }
      // Break early if both circles are found
      if (startCircle && endCircle)
      {
        break;
      }
    }

    if (!startCircle || !endCircle)
    {
      LOGERR << "One or both circles not found. Fitness cannot be computed." << std::endl;
      m_fObjectiveFunction = 0.0f;
      return;
    }

    // Step 1: Sample points along the line connecting both circles
    std::vector<CVector2> samplePoints;
    CVector2 startPos = startCircle->center; // Use the center attribute
    CVector2 endPos = endCircle->center;     // Use the center attribute
    Real lineLength = (endPos - startPos).Length();
    size_t numSamples = static_cast<size_t>(lineLength / connectionRange);

    for (size_t i = 0; i <= numSamples; ++i)
    {
      Real t = static_cast<Real>(i) / static_cast<Real>(numSamples);
      CVector2 samplePoint = startPos + t * (endPos - startPos);
      samplePoints.emplace_back(samplePoint);
    }

    // Step 2: Initialize data structures
    std::map<size_t, std::vector<CEPuckEntity *>> closestRobotsMap; // Change key type to size_t
    Real totalClosestDistance = 0.0f;

    // Step 3: Enumerate robots
    CSpace::TMapPerType &tEpuckMap = GetSpace().GetEntitiesByType("epuck");
    for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
    {
      CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);
      CVector2 robotPosition(pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),
                             pcEpuck->GetEmbodiedEntity().GetOriginAnchor().Position.GetY());

      Real closestDistance = std::numeric_limits<Real>::max();
      size_t closestSampleIndex = 0; // Store the index of the closest sample point

      // Step 4: Find closest sample point
      for (size_t i = 0; i < samplePoints.size(); ++i)
      { // Use index i
        const CVector2 &samplePoint = samplePoints[i];
        Real distance = (robotPosition - samplePoint).Length();
        if (distance < closestDistance)
        {
          closestDistance = distance;
          closestSampleIndex = i; // Update the index of the closest sample point
        }
      }

      // Step 5: Store closest robot information using the index
      closestRobotsMap[closestSampleIndex].push_back(pcEpuck);
      totalClosestDistance += closestDistance;
    }

    // Step 6: Count robots with multiple neighbors
    int countOfRobotsWithMultipleNeighbors = 0;
    for (const auto &entry : closestRobotsMap)
    {
      if (entry.second.size() >= 2)
      {
        countOfRobotsWithMultipleNeighbors += entry.second.size();
      }
    }

    // Step 7: Output results
    std::cout << "Total Closest Distance: " << totalClosestDistance << std::endl;
    std::cout << "Number of Robots with Multiple Neighbors: " << countOfRobotsWithMultipleNeighbors << std::endl;

    // Store or print the fitness value
    m_fObjectiveFunction += -totalClosestDistance - countOfRobotsWithMultipleNeighbors / 10.0f; // just use most recent result
    std::cout << "Total Fitness (neg sum of distances to closest robots): " << m_fObjectiveFunction << std::endl;
  }

  // Create and open a text file
  ofstream MyFile("pos.mu", ios::trunc);

  CSpace::TMapPerType &tEpuckMap = GetSpace().GetEntitiesByType("epuck");
  CVector2 cEpuckPosition(0, 0);

  for (CSpace::TMapPerType::iterator it = tEpuckMap.begin(); it != tEpuckMap.end(); ++it)
  {
    CEPuckEntity *pcEpuck = any_cast<CEPuckEntity *>(it->second);
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

Real Template::GetObjectiveFunction()
{
  return m_fObjectiveFunction;
}

/****************************************/
/****************************************/

CVector3 Template::GetRandomPosition()
{

  int lenQuad = initRectangles.size();
  int lenDisk = initCircles.size();
  int randArea = m_pcRng->Uniform(CRange<int>(1, lenDisk + lenQuad));
  int ind = 0;

  LOG << "epucks left before " << epucks.size() << endl;
  if (epucks.size() > 0)
  {
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
    advance(it, area - 1);
    Rectangle rectArea = *it;

    Real a = m_pcRng->Uniform(CRange<Real>(-1.0f, 1.0f));
    Real b = m_pcRng->Uniform(CRange<Real>(-1.0f, 1.0f));

    Real theta = -rectArea.angle * (M_PI / 180);

    Real posX = rectArea.center.GetX() + a * rectArea.width * cos(theta) - b * rectArea.height * sin(theta);
    Real posY = rectArea.center.GetY() + a * rectArea.width * sin(theta) + b * rectArea.height * cos(theta);

    return CVector3(posX, posY, 0);
  }
  else
  {
    int area = randArea;
    std::list<Circle>::iterator it = initCircles.begin();
    advance(it, area - 1);
    Circle diskArea = *it;

    Real temp;
    Real a = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
    Real b = m_pcRng->Uniform(CRange<Real>(0.0f, 1.0f));
    // If b < a, swap them
    if (b < a)
    {
      temp = a;
      a = b;
      b = temp;
    }
    Real posX = diskArea.center.GetX() + b * diskArea.radius * cos(2 * CRadians::PI.GetValue() * (a / b));
    Real posY = diskArea.center.GetY() + b * diskArea.radius * sin(2 * CRadians::PI.GetValue() * (a / b));

    return CVector3(posX, posY, 0);
  }
}

REGISTER_LOOP_FUNCTIONS(Template, "template");
