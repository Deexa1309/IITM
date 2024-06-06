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

// Incremental update for a new node
void incrementalPageRankUpdate(unordered_map<int, vector<int>>& adjList, vector<int>& outdegree, vector<double>& rank, int newNode, const vector<int>& neighbors) {
    adjList[newNode] = neighbors;
    outdegree[newNode] = neighbors.size();

    vector<double> newRank(rank.size(), (1.0 - DAMPING_FACTOR) / rank.size());

    bool converged;
    do {
        #pragma omp parallel
        {
            vector<double> localNewRank(rank.size(), 0.0);

            #pragma omp for
            for (int i = 0; i < rank.size(); ++i) {
                if (outdegree[i] > 0) {
                    double contribution = DAMPING_FACTOR * rank[i] / outdegree[i];
                    for (int neighbor : adjList[i]) {
                        localNewRank[neighbor] += contribution;
                    }
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < rank.size(); ++i) {
                    newRank[i] += localNewRank[i];
                }
            }
        }

        converged = isConverged(rank, newRank);

        #pragma omp parallel for
        for (int i = 0; i < rank.size(); ++i) {
            rank[i] = newRank[i];
        }

    } while (!converged);
}

int main() {
    int numThreads;
    cout << "\nEnter number of threads: ";
    cin >> numThreads;
    omp_set_num_threads(numThreads);

    string filename;
    cout << "\n Enter the filename of the graph: ";
    cin >> filename;

    unordered_map<int, vector<int>> adjList = loadGraph(filename);

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

    // Adding a new node
    int newNode;
    cout << "Enter the new node ID: ";
    cin >> newNode;

    int numNeighbors;
    cout << "Enter the number of neighbors for the new node: ";
    cin >> numNeighbors;

    vector<int> neighbors(numNeighbors);
    cout << "Enter the neighbors of the new node: ";
    for (int i = 0; i < numNeighbors; ++i) {
        cin >> neighbors[i];
    }

    auto updateStart = chrono::high_resolution_clock::now();
    incrementalPageRankUpdate(adjList, outdegree, rank, newNode, neighbors);
    auto updateEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> updateElapsed = updateEnd - updateStart;

    cout << "Updated PageRank values after adding the new node." << endl;

    nodeRankPairs.resize(NUM_NODES + 1);
    for (int i = 0; i < NUM_NODES + 1; ++i) {
        nodeRankPairs[i] = {i, rank[i]};
    }

    sort(nodeRankPairs.begin(), nodeRankPairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return b.second < a.second;
    });

    cout << "Top 10 nodes by updated PageRank:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << "Node " << nodeRankPairs[i].first << ": " << nodeRankPairs[i].second << endl;
    }

    cout << "Update execution time: " << updateElapsed.count() << " seconds" << endl;

    return 0;
}

