#include "engine/engine.h"
#include "utility/optionparser.h"

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace CityFlow;

int main(int argc, char const *argv[]) {
    optionparser::OptionParser parser;

    parser.add_option("--configFile", "-c")
            .help("config file")
            .mode(optionparser::StorageMode::STORE_VALUE)
            .required(true);

    parser.add_option("--totalStep", "-s")
            .help("simulation steps")
            .default_value(1000)
            .mode(optionparser::StorageMode::STORE_VALUE);

    parser.add_option("--threadNum", "-t")
            .help("number of threads")
            .default_value(1)
            .mode(optionparser::StorageMode::STORE_VALUE);

    parser.add_option("--verbose", "-v")
            .help("be verbose")
            .mode(optionparser::StorageMode::STORE_TRUE);

    parser.eat_arguments(argc, argv);
    std::string configFile = parser.get_value<std::string>("configFile");
    bool verbose = parser.get_value("verbose");
    size_t totalStep = parser.get_value<int>("totalStep");
    size_t threadNum = parser.get_value<int>("threadNum");

    std::string dataDir(std::getenv("DATADIR"));
//    std::string dataDir = "/Users/dongxing/Documents/04_program/trafficLight/CityFlow-master/";
//
//    Engine engine(dataDir + configFile, (size_t) threadNum);
//    engine.reset();
//    engine.setTrafficLightPhase("intersection_1_1", 0);
//    engine.setTrafficLightPhase("intersection_1_2", 0);
//    engine.setTrafficLightPhase("intersection_1_3", 0);
//    engine.setTrafficLightPhase("intersection_2_1", 0);
//    engine.setTrafficLightPhase("intersection_2_2", 0);
//    engine.setTrafficLightPhase("intersection_2_3", 0);
//    engine.setTrafficLightPhase("intersection_3_1", 0);
//    engine.setTrafficLightPhase("intersection_3_2", 0);
//    engine.setTrafficLightPhase("intersection_3_3", 0);
//    engine.setTrafficLightPhase("intersection_4_1", 0);
//    engine.setTrafficLightPhase("intersection_4_2", 0);
//    engine.setTrafficLightPhase("intersection_4_3", 0);
//
//    std::map<std::string, int> lane2vehicle = engine.getLaneVehicleCount();
//    for (const auto& lane : lane2vehicle) {
//        std::cout << lane.first << "\t" << lane.second << std::endl;
//    }
//
//    for (int i = 0; i < 3; ++i) {
//        engine.nextStep();
//    }
//    engine.setTrafficLightPhase("intersection_1_1", 1);
//    engine.setTrafficLightPhase("intersection_1_2", 1);
//    engine.setTrafficLightPhase("intersection_1_3", 1);
//    engine.setTrafficLightPhase("intersection_2_1", 1);
//    engine.setTrafficLightPhase("intersection_2_2", 1);
//    engine.setTrafficLightPhase("intersection_2_3", 1);
//    engine.setTrafficLightPhase("intersection_3_1", 1);
//    engine.setTrafficLightPhase("intersection_3_2", 1);
//    engine.setTrafficLightPhase("intersection_3_3", 1);
//    engine.setTrafficLightPhase("intersection_4_1", 0);
//    engine.setTrafficLightPhase("intersection_4_2", 1);
//    engine.setTrafficLightPhase("intersection_4_3", 1);
//    for (int i = 0; i < 17; ++i) {
//        engine.nextStep();
//    }

    time_t startTime, endTime;
    time(&startTime);
    for (int i = 0; i < totalStep; i++) {
        if (verbose) {
            std::cout << i << " " << engine.getVehicleCount() << std::endl;

            std::map<std::string, int> lane2vehicle = engine.getLaneVehicleCount();
            for (const auto& lane : lane2vehicle) {
                std::cout << lane.first << "\t" << lane.second << std::endl;
            }
        }
        engine.nextStep();
        //engine.getVehicleSpeed();
        //engine.getLaneVehicles();
        //engine.getLaneWaitingVehicleCount();
        //engine.getVehicleDistance();
        //engine.getCurrentTime();
    }
    time(&endTime);
    std::cout << "Total Step: " << totalStep << std::endl;
    std::cout << "Total Time: " << (endTime - startTime) << "s" << std::endl;
    return 0;
}