//author:wxb.tju@gmail.com

#include "deformationGraph.h"
#include "stdio.h"

using namespace cv;

void m_sort(int *p_ind, float *p_dist, float *dist, int num)
{
    int current_ind, i, j;
    int k = 4;
    float curr_dist, max_dist;
    int   max_row;
    p_dist[0] = dist[0];p_dist[1] = dist[1];p_dist[2] = dist[2];p_dist[3] = dist[3];
    max_dist = p_dist[0];
    p_ind[0] = 1;
    for (current_ind=1; current_ind<k; current_ind++)
    {
        curr_dist = p_dist[current_ind];
        if (curr_dist<max_dist)
        {
            i=current_ind-1;
            for (int a=0; a<current_ind-1; a++)
            {
                if (p_dist[a]>curr_dist)
                {
                    i=a;
                    break;
                  }
              }
            for (j=current_ind; j>i; j--)
            {
                p_dist[j] = p_dist[j-1];
                p_ind[j]   = p_ind[j-1];
              }
            p_dist[i] = curr_dist;
            p_ind[i]   = current_ind+1;
          }
        else
        {
          p_ind[current_ind] = current_ind+1;
        }
        max_dist = p_dist[current_ind];
      }

        // Part 2 : insert element in the k-th first lines
        max_row = k-1;
        for (current_ind=k; current_ind<num; current_ind++)
        {
            curr_dist = dist[current_ind];
            if (curr_dist<max_dist){
                i=k-1;
                for (int a=0; a<k-1; a++){
                    if (p_dist[a]>curr_dist){
                        i=a;
                        break;
                      }
                  }
                for (j=k-1; j>i; j--){
                    p_dist[j] = p_dist[j-1];
                    p_ind[j]   = p_ind[j-1];
                  }
                p_dist[i] = curr_dist;
                p_ind[i]  = current_ind+1;
                max_dist  = p_dist[max_row];
              }
          }
}

DeformationGraph::DeformationGraph()
    :idx(0),
      noMaxNodes(1000)
{}

DeformationGraph::~DeformationGraph()
{
}

void DeformationGraph::updateNodes(const Image<int> & knn_map, const Image<float3> & vertex, cv::Mat & img, const Matrix4 & pose, float dis_thrsh)
{
//    vector<ITMNodes::Node> tempNode;
    for(int i=0;i<vertex.size.x;i++)
      for(int j=0;j<vertex.size.y;j++)
      {
          if(knn_map[make_uint2(i,j)] == 1 && graphNodes.size() < noMaxNodes)
          {
              if(0 == graphNodes.size())
              {
                  GraphNode tempNode;
                  tempNode.id = 0;
//                  tempNode.g_v[0] = vertex[make_uint2(i,j)].x; tempNode.g_v[1] = vertex[make_uint2(i,j)].y;tempNode.g_v[2] = vertex[make_uint2(i,j)].z;
                  tempNode.g_v = Tbx::Point3(vertex[make_uint2(i,j)].x, vertex[make_uint2(i,j)].y, vertex[make_uint2(i,j)].z);
                  tempNode.g_w = dis_thrsh;
                  Tbx::Transfo trans(pose.data[0].x, pose.data[0].y, pose.data[0].z, pose.data[0].w,
                                                  pose.data[1].x, pose.data[1].y, pose.data[1].z, pose.data[1].w,
                                                  pose.data[2].x, pose.data[2].y, pose.data[2].z, pose.data[2].w,
                                                  pose.data[3].x, pose.data[3].y, pose.data[3].z, pose.data[3].w);
                  tempNode.g_dq = Tbx::Dual_quat_cu(trans);
                  graphNodes.push_back(tempNode);
                  idx++;
              }
              else
              {
                  float dist = 0;
                  for(int k=0;k<graphNodes.size();k++)
                  {
//                      dist = (vertex[make_uint2(i,j)].x - graphNodes[k].g_v[0])*(vertex[make_uint2(i,j)].x - graphNodes[k].g_v[0])
//                          + (vertex[make_uint2(i,j)].y - graphNodes[k].g_v[1])*(vertex[make_uint2(i,j)].y - graphNodes[k].g_v[1])
//                          + (vertex[make_uint2(i,j)].z - graphNodes[k].g_v[2])*(vertex[make_uint2(i,j)].z - graphNodes[k].g_v[2]);
                      dist = (vertex[make_uint2(i,j)].x - graphNodes[k].g_v.x)*(vertex[make_uint2(i,j)].x - graphNodes[k].g_v.x)
                          + (vertex[make_uint2(i,j)].y - graphNodes[k].g_v.y)*(vertex[make_uint2(i,j)].y - graphNodes[k].g_v.y)
                          + (vertex[make_uint2(i,j)].z - graphNodes[k].g_v.z)*(vertex[make_uint2(i,j)].z - graphNodes[k].g_v.z);
                      if(dist < dis_thrsh*dis_thrsh)         //0.2 is dist threshold
                      {
                          break;
                      }
                  }
                  if(dist > dis_thrsh*dis_thrsh && graphNodes.size() < noMaxNodes)
                  {
                      GraphNode tempNode;
                      tempNode.id = idx;
//                      tempNode.g_v[0] = vertex[make_uint2(i,j)].x; tempNode.g_v[1] = vertex[make_uint2(i,j)].y;tempNode.g_v[2] = vertex[make_uint2(i,j)].z;
                      tempNode.g_v = Tbx::Point3(vertex[make_uint2(i,j)].x, vertex[make_uint2(i,j)].y, vertex[make_uint2(i,j)].z);
                      tempNode.g_w = dis_thrsh;
                      Tbx::Transfo trans(pose.data[0].x, pose.data[0].y, pose.data[0].z, pose.data[0].w,
                                                      pose.data[1].x, pose.data[1].y, pose.data[1].z, pose.data[1].w,
                                                      pose.data[2].x, pose.data[2].y, pose.data[2].z, pose.data[2].w,
                                                      pose.data[3].x, pose.data[3].y, pose.data[3].z, pose.data[3].w);
                      tempNode.g_dq = Tbx::Dual_quat_cu(trans);
                      graphNodes.push_back(tempNode);
//                      cv::circle(img, Point(i, j), 2, CV_RGB(0, 255, 0), 3);
                      node_points.push_back(Point(i, j));
                      idx++;
                  }
              }
          }
      }

    updateRegularization(dis_thrsh, img);
//    float d[10];
//    d[0] = 0.8;d[1] = 5.4;d[2] = 2.3;d[3] = 2.4;d[4] = 0.9;d[5] = 2.5;d[6] = 1.3;d[7] = 0.4;d[8] = 4.3;d[9] = 1.4;
//    int p_ind[4];
//    float p_dist[4];
//    m_sort(p_ind, p_dist, d, 10);
//    std::cout<<p_dist[0]<<" "<<p_dist[1]<<" "<<p_dist[2]<<" "<<p_dist[3]<<std::endl;
//    std::cout<<p_ind[0]-1<<" "<<p_ind[1]-1<<" "<<p_ind[2]-1<<" "<<p_ind[3]-1<<std::endl;
}

void DeformationGraph::updateRegularization(float dis_thrsh, cv::Mat & img)
{
    //construct regNodes
    float dis_reg1 = 4 * dis_thrsh;
    float dis_reg2 = 4 * dis_reg1;
    std::vector<int> graphRegularization1;
    std::vector<int> graphRegularization2;
    graphRegularization1.push_back(0);
    graphRegularization2.push_back(0);
    for(int i=1;i<graphNodes.size();i++)
    {
        float dist = 0.0;
        for(int j=0;j<graphRegularization1.size();j++)
        {
            dist = (graphNodes[i].g_v.x - graphNodes[graphRegularization1[j]].g_v.x)*(graphNodes[i].g_v.x - graphNodes[graphRegularization1[j]].g_v.x)
                        + (graphNodes[i].g_v.y - graphNodes[graphRegularization1[j]].g_v.y)*(graphNodes[i].g_v.y - graphNodes[graphRegularization1[j]].g_v.y)
                        + (graphNodes[i].g_v.z - graphNodes[graphRegularization1[j]].g_v.z)*(graphNodes[i].g_v.z - graphNodes[graphRegularization1[j]].g_v.z);

            if(dist < dis_reg1*dis_reg1)
            {
                break;
            }
        }
        if(dist > dis_reg1*dis_reg1)
        {
            graphRegularization1.push_back(i);
        }
    }

    for(int i=1;i<graphNodes.size();i++)
    {
        float dist = 0.0;
        for(int j=0;j<graphRegularization2.size();j++)
        {
            dist = (graphNodes[i].g_v.x - graphNodes[graphRegularization2[j]].g_v.x)*(graphNodes[i].g_v.x - graphNodes[graphRegularization2[j]].g_v.x)
                        + (graphNodes[i].g_v.y - graphNodes[graphRegularization2[j]].g_v.y)*(graphNodes[i].g_v.y - graphNodes[graphRegularization2[j]].g_v.y)
                        + (graphNodes[i].g_v.z - graphNodes[graphRegularization2[j]].g_v.z)*(graphNodes[i].g_v.z - graphNodes[graphRegularization2[j]].g_v.z);

            if(dist < dis_reg2*dis_reg2)
            {
                break;
            }
        }
        if(dist > dis_reg2*dis_reg2)
        {
            graphRegularization2.push_back(i);
        }
    }

    if(graphRegularization1.size() >= 4 && graphRegularization2.size() >= 4)
    {
        //knn
        for(int m=0;m<graphNodes.size();m++)
        {
            int num = graphRegularization1.size();
            float dist[num];
            for(int n=0;n<num;n++)
            {
                dist[n] = sqrt((graphNodes[m].g_v.x - graphNodes[graphRegularization1[n]].g_v.x)*(graphNodes[m].g_v.x - graphNodes[graphRegularization1[n]].g_v.x)
                                    + (graphNodes[m].g_v.y - graphNodes[graphRegularization1[n]].g_v.y)*(graphNodes[m].g_v.y - graphNodes[graphRegularization1[n]].g_v.y)
                                    + (graphNodes[m].g_v.z - graphNodes[graphRegularization1[n]].g_v.z)*(graphNodes[m].g_v.z - graphNodes[graphRegularization1[n]].g_v.z));
            }
            int p_ind[4];
            float p_dist[4];
            m_sort(p_ind, p_dist, dist, num);
            graphNodes[m].reg_dist[0] = p_dist[0]; graphNodes[m].reg_dist[1] = p_dist[1]; graphNodes[m].reg_dist[2] = p_dist[2]; graphNodes[m].reg_dist[3] = p_dist[3];
            graphNodes[m].reg_ind[0] = graphRegularization1[p_ind[0]-1]; graphNodes[m].reg_ind[1] = graphRegularization1[p_ind[1]-1]; graphNodes[m].reg_ind[2] = graphRegularization1[p_ind[2]-1]; graphNodes[m].reg_ind[3] = graphRegularization1[p_ind[3]-1];
            graphNodes[m].IsReg = false;
        }

        //knn
        for(int m=0;m<graphRegularization1.size();m++)
        {
            int num = graphRegularization2.size();
            float dist[num];
            for(int n=0;n<num;n++)
            {
                dist[n] = sqrt((graphNodes[graphRegularization1[m]].g_v.x - graphNodes[graphRegularization2[n]].g_v.x)*(graphNodes[graphRegularization1[m]].g_v.x - graphNodes[graphRegularization2[n]].g_v.x)
                                    + (graphNodes[graphRegularization1[m]].g_v.y - graphNodes[graphRegularization2[n]].g_v.y)*(graphNodes[graphRegularization1[m]].g_v.y - graphNodes[graphRegularization2[n]].g_v.y)
                                    + (graphNodes[graphRegularization1[m]].g_v.z - graphNodes[graphRegularization2[n]].g_v.z)*(graphNodes[graphRegularization1[m]].g_v.z - graphNodes[graphRegularization2[n]].g_v.z));

                cv::circle(img, node_points[graphRegularization2[n]], 2, CV_RGB(0, 0, 255), 3);
            }
            int p_ind[4];
            float p_dist[4];
            m_sort(p_ind, p_dist, dist, num);

            graphNodes[graphRegularization1[m]].reg_dist2[0] = p_dist[0]; graphNodes[graphRegularization1[m]].reg_dist2[1] = p_dist[1]; graphNodes[graphRegularization1[m]].reg_dist2[2] = p_dist[2]; graphNodes[graphRegularization1[m]].reg_dist2[3] = p_dist[3];
            graphNodes[graphRegularization1[m]].reg_ind2[0] = graphRegularization2[p_ind[0]-1]; graphNodes[graphRegularization1[m]].reg_ind2[1] = graphRegularization2[p_ind[1]-1]; graphNodes[graphRegularization1[m]].reg_ind2[2] = graphRegularization2[p_ind[2]-1]; graphNodes[graphRegularization1[m]].reg_ind2[3] = graphRegularization2[p_ind[3]-1];
            graphNodes[graphRegularization1[m]].IsReg = true;

//            std::cout<<graphNodes[graphRegularization1[m]].reg_dist2[0]<<" "<<graphNodes[graphRegularization1[m]].reg_dist2[1]<<" "<<graphNodes[graphRegularization1[m]].reg_dist2[2]<<" "<<graphNodes[graphRegularization1[m]].reg_dist2[3]<<std::endl;
//            std::cout<< graphNodes[graphRegularization1[m]].reg_ind2[0]<<" "<<graphNodes[graphRegularization1[m]].reg_ind2[1]<<" "<< graphNodes[graphRegularization1[m]].reg_ind2[2]<<" "<< graphNodes[graphRegularization1[m]].reg_ind2[3]<<std::endl;
//            std::cout<<m << " "<< num<< " "<< graphRegularization2[0]<<endl;

            cv::circle(img, node_points[graphRegularization1[m]], 2, CV_RGB(0, 255, 0), 3);
            if(m == 50)
            {
                cv::line(img, node_points[graphRegularization1[m]], node_points[graphNodes[graphRegularization1[m]].reg_ind2[0]], CV_RGB(255, 0, 0), 1);
                cv::line(img, node_points[graphRegularization1[m]], node_points[graphNodes[graphRegularization1[m]].reg_ind2[1]], CV_RGB(255, 0, 0), 1);
                cv::line(img, node_points[graphRegularization1[m]], node_points[graphNodes[graphRegularization1[m]].reg_ind2[2]], CV_RGB(255, 0, 0), 1);
                cv::line(img, node_points[graphRegularization1[m]], node_points[graphNodes[graphRegularization1[m]].reg_ind2[3]], CV_RGB(255, 0, 0), 1);
            }
        }
    }

    imshow("1", img);
    waitKey(1);
}
