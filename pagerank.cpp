#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <algorithm>

using namespace std;

const int NUM_NODES = 1000000; 
const double DAMPING_FACTOR = 0.85;
const double EPSILON = 1e-6;

// convergence
bool isConverged(const vector<double>& oldRank, const vector<double>& newRank) {
    for (int i = 0; i < oldRank.size(); ++i) {
        if (fabs(oldRank[i] - newRank[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

// Graph loading
unordered_map<int, vector<int>> loadGraph(const string& filename) {
    unordered_map<int, vector<int>> adjList;
    ifstream graphFile(filename);
    string line;
    while (getline(graphFile, line)) {
        istringstream iss(line);
        int source, destination;
        if (iss >> source >> destination) {
            adjList[source].push_back(destination);
        }
    }
    graphFile.close();
    return adjList;
}

int main() {
    int numThreads;
    cout << "\nEnter number of threads: ";
    cin >> numThreads;
    omp_set_num_threads(numThreads);

    unordered_map<int, vector<int>> adjList = loadGraph("graph.txt");

    vector<int> outdegree(NUM_NODES, 0);
    for (const auto& pair : adjList) {
        int node = pair.first;
        outdegree[node] = pair.second.size();
    }

    vector<double> rank(NUM_NODES, 1.0 / NUM_NODES);
    vector<double> newRank(NUM_NODES, 0.0);

    auto start = chrono::high_resolution_clock::now();

    bool converged;
    do {
        #pragma omp parallel for
        for (int i = 0; i < NUM_NODES; ++i) {
            newRank[i] = (1.0 - DAMPING_FACTOR) / NUM_NODES;
        }

        #pragma omp parallel
        {
            vector<double> localNewRank(NUM_NODES, 0.0);

            #pragma omp for
            for (int i = 0; i < NUM_NODES; ++i) {
                if (outdegree[i] > 0) {
                    double contribution = DAMPING_FACTOR * rank[i] / outdegree[i];
                    for (int neighbor : adjList[i]) {
                        localNewRank[neighbor] += contribution;
                    }
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < NUM_NODES; ++i) {
                    newRank[i] += localNewRank[i];
                }
            }
        }

        converged = isConverged(rank, newRank);

        #pragma omp parallel for
        for (int i = 0; i < NUM_NODES; ++i) {
            rank[i] = newRank[i];
        }

    } while (!converged);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    vector<pair<int, double>> nodeRankPairs(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) {
        nodeRankPairs[i] = {i, rank[i]};
    }

  
    sort(nodeRankPairs.begin(), nodeRankPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return b.second < a.second;
    });

   
    cout << "Top 10 nodes by PageRank:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << "Node " << nodeRankPairs[i].first << ": " << nodeRankPairs[i].second << endl;
    }

    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}

