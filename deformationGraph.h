//author:wxb.tju@gmail.com

#include <vector>

#include "fusion.h"
#include "graphNode.h"
#include "opencv2/opencv.hpp"

using namespace std;

class DeformationGraph
{
    public:
        DeformationGraph();
        virtual ~DeformationGraph();

        std::vector<GraphNode> graphNodes;

        void updateNodes(const Image<int> & knn_map, const Image<float3> & vertex, cv::Mat & img, const Matrix4 & pose, float dis_thrsh = 0.1);

        void updateRegularization(float dis_thrsh, cv::Mat & img);

    private:
        int idx;
        int noMaxNodes;

        std::vector <cv::Point> node_points;
};
