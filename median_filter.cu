extern "C"{
    __device__ float find_median_in_one_dim_array(float* neighb_array, const int N)
    {
        int i, j;
        float key;

        for (i = 1; i < N; i++)
        {
            key = neighb_array[i];
            j = i - 1;

            while (j >= 0 && neighb_array[j] > key)
            {
                neighb_array[j + 1] = neighb_array[j];
                j = j - 1;
            }
            neighb_array[j + 1] = key;
        }
        return neighb_array[N / 2];
    }
    __device__ void print_neighbour_array(const float* array, const int N)
    {
        for (int i = 0; i < N; i++)
            printf("%.6f ", array[i]);
        printf("\n");
    }
    __device__ float find_neighbour_median(const float* padded_array, const int index_offset, const int padded_img_width, const int id_x, const int id_y, const int filter_size)
    {
        float neighb_array[25];
        int n_counter = 0;

        for (int i = id_x; i < id_x + filter_size; i++)
        {
            for (int j = id_y; j < id_y + filter_size; j++)
            {
                neighb_array[n_counter] = padded_array[index_offset + (i * padded_img_width) + j];
                n_counter += 1;
            }
        }

        return find_median_in_one_dim_array(neighb_array, filter_size * filter_size);
    }
    __global__ void three_dim_median_filter(float* data_array, const float* padded_array, const int N_IMAGES, const int X, const int Y, const int filter_size)
    {
        unsigned int id_img = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int id_x = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned int id_y = blockIdx.z*blockDim.z + threadIdx.z;
        unsigned int img_size =  X * Y;
        unsigned int padded_img_width =  X + filter_size - 1;
        unsigned int padded_img_size =  padded_img_width * (Y + filter_size - 1);

        if ((id_img >= N_IMAGES) || (id_x >= X) || (id_y >= Y))
            return;

        data_array[(id_img * img_size) + (id_x * X) + id_y] = find_neighbour_median(padded_array, id_img * padded_img_size, padded_img_width, id_x, id_y, filter_size);
    }
    __global__ void two_dim_median_filter(float* data_array, const float* padded_array, const int X, const int Y, const int filter_size)
    {
        unsigned int id_x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int id_y = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned int padded_img_width =  X + filter_size - 1;

        if ((id_x >= X) || (id_y >= Y))
            return;

        data_array[(id_x * X) + id_y] = find_neighbour_median(padded_array, 0, padded_img_width, id_x, id_y, filter_size);  
    }
    __global__ void two_dim_remove_light_outliers(float* data_array, const float* padded_array, const int X, const int Y, const int filter_size, const float diff)
    {
        unsigned int id_x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int id_y = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned int index = (id_x * X) + id_y;
        unsigned int padded_img_width =  X + filter_size - 1;

        if ((id_x >= X) || (id_y >= Y))
            return;

        float median = find_neighbour_median(padded_array, 0, padded_img_width, id_x, id_y, filter_size);

        if (data_array[index] - median >= diff)
            data_array[index] = median;
    }
    __global__ void two_dim_remove_dark_outliers(float* data_array, const float* padded_array, const int X, const int Y, const int filter_size, const float diff)
    {
        unsigned int id_x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int id_y = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned int index = (id_x * X) + id_y;
        unsigned int padded_img_width =  X + filter_size - 1;

        if ((id_x >= X) || (id_y >= Y))
            return;

        float median = find_neighbour_median(padded_array, 0, padded_img_width, id_x, id_y, filter_size);

        if (median - data_array[index] >= diff)
            data_array[index] = median;
    }
}
