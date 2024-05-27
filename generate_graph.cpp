#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

const int NUM_NODES = 1000000; 
const int AVG_FOLLOWERS = 50; 

int main() {
    
    srand(time(0));

  
    ofstream graphFile("graph.txt");

    // Randomly generate graph edges to simulate a Twitter-like follower network
    for (int i = 0; i < NUM_NODES; ++i) {
        int numFollowers = rand() % (2 * AVG_FOLLOWERS); 
        for (int j = 0; j < numFollowers; ++j) {
            int follower = rand() % NUM_NODES;
            if (follower != i) {
                graphFile << i << " " << follower << endl;
            }
        }
    }

    graphFile.close();
    cout << "Graph generated and saved to graph.txt" << endl;
    return 0;
}

