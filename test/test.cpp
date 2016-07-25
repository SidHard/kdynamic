#include <iostream>

int main()
{
  int current_ind, i, j;
  int k = 4;
  float p_dist[4];
  float d[10];
  int   p_ind[4];
  float curr_dist, max_dist;
  int   max_row;

	
	d[0] = 0.2;d[1] = 5.4;d[2] = 2.3;d[3] = 2.4;d[4] = 0.9;d[5] = 2.5;d[6] = 1.3;d[7] = 0.4;d[8] = 4.3;d[9] = 1.4;
	p_dist[0] = d[0];p_dist[1] = d[1];p_dist[2] = d[2];p_dist[3] = d[3];

	max_dist = p_dist[0];
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
      for (current_ind=k; current_ind<10; current_ind++)
      {
          curr_dist = d[current_ind];

    //      __syncthreads();

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

	std::cout<<p_dist[0]<<" "<<p_dist[1]<<" "<<p_dist[2]<<" "<<p_dist[3]<<std::endl;
	std::cout<<p_ind[0]<<" "<<p_ind[1]<<" "<<p_ind[2]<<" "<<p_ind[3]<<std::endl;
}
