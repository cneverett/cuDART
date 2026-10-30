#ifndef MESHBLOCK_HPP_
#define MESHBLOCK_HPP_

#include <stdexcept>

#include "vec3.hpp"
#include "ray.hpp"
#include "tools.hpp"
#include "npy.hpp"

struct MeshBlockInfo {
    int mb_size, mem_start, mb_index;
    bool beta_in_data;
    vec3 mb_dims, xl, xr;
    // flexload data
    vec3 mb_origin;
    float mb_radius;
};

class MeshBlock {
    // a MeshBlock provides a wrapper for data that allows for tracing by rays
    // MesHBlocks are purely device objects, including a pointer to traceable data device-allocated externally 
    public:
        // ctors
        __device__ MeshBlock() {}
        __device__ MeshBlock(MeshBlockInfo mb_info, float *all_data);

        // routines
        __device__ bool calc_mb_intercept(const Ray &r, float &s_entry, float &s_exit);
        __device__ int int_clamp(float f, float l, float r);
        __device__ vec3 get_edge(bool sign) {return (sign) ? xr : xl;}
        __device__ void print_data();
        __device__ float calc_trace(const Ray &r, TraceArgs trace_args);

        // public properties
        const int axes_bitmap[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        float *all_data;
        float sum;
        int mb_size, mem_start;
        vec3 xl, xr, dx, mb_dims;
        bool beta_in_data;
};

__host__ std::vector<MeshBlockInfo> load_unlabelled_meshblock(std::string input_str, float* &h_all_data, size_t &h_bytes, bool relativistic, bool verbose, bool host_malloc) {
    // load single homgenous meshblock info, allocate host memory and load data

    // start timer for load
    auto npy_read_start = std::chrono::steady_clock::now();

    std::vector<MeshBlockInfo> all_mb_info = {};
    bool file_exists = std::filesystem::is_regular_file(input_str);
    if (!file_exists) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in meshblock.hpp ###\n";
        err_msg << "Unable to locate input file at " << input_str << std::endl;
        CUDART_ERROR(err_msg);
    }
    std::vector<unsigned long int> data_shape = npy_to_host(input_str, h_all_data, h_bytes, verbose, host_malloc);
    vec3 mb_dims((float)data_shape[0], (float)data_shape[1], (float)data_shape[2]); // extract first three indices for spatial dims
    int mb_size = data_shape[0] * data_shape[1] * data_shape[2]; // number of spatial cells (<= all cells)
    bool beta_in_data = (data_shape.size() > 3); // if fourth axis, contains velocity data
    // check dimensions
    if ((data_shape.size() == 3) && (relativistic)) {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in meshblock.hpp ###\n";
        err_msg << "Relativistic flagged, but did not find velocity data in " << input_str << std::endl;
        CUDART_ERROR(err_msg);
    }  
    if (verbose) {
        std::chrono::duration<float> npy_read_dur = (std::chrono::steady_clock::now() - npy_read_start);
        printf("npy read                  (host)              %.6fs\n",npy_read_dur.count());
    }

    //assume equal spacing in x, y, z and centering at origin
    float longest_side = static_cast<float>(*std::max_element(data_shape.begin(), data_shape.end()));
    vec3 mb_extent = mb_dims / longest_side;
    vec3 xl = -0.5 * mb_extent;
    vec3 xr = 0.5 * mb_extent;

    // stash info
    MeshBlockInfo mb_info;
    mb_info.mb_size = mb_size;
    mb_info.xl = xl;
    mb_info.xr = xr;
    mb_info.mb_dims = mb_dims;
    mb_info.beta_in_data = beta_in_data;
    mb_info.mem_start = 0;
    mb_info.mb_index = 0;
    // flexload data
    mb_info.mb_origin = 0.5 * (mb_info.xl + mb_info.xr);
    mb_info.mb_radius = (mb_info.xl - mb_info.mb_origin).vector_mag();
    all_mb_info.push_back(mb_info);

    return all_mb_info;
}

__host__ std::vector<MeshBlockInfo> load_labelled_meshblocks(std::string input_str, float* &h_all_data, size_t &h_bytes, bool relativistic, bool verbose, bool host_malloc) {
    // load heterogeneous meshblock info, allocate host memory and load data

    // start timer for header read
    auto header_init_start = std::chrono::steady_clock::now();

    // read header data
    std::vector<MeshBlockInfo> all_mb_info = {};
    std::string header_str = input_str + "/header.txt";
    std::ifstream header_file(header_str);
    int npy_floats = 0;
    if (header_file.is_open()) {
        std::string line;
        int line_count = 0;
        int mb_size, nx, ny, nz;
        float xl, yl, zl, xr, yr, zr;
        while (std::getline(header_file, line)) {
            std::istringstream iss(line);
            if (!(iss >> mb_size >> nx >> ny >> nz >> xl >> yl >> zl >> xr >> yr >> zr)) {
                std::stringstream err_msg;
                err_msg << "### FATAL ERROR in meshblock.hpp ###\n";
                err_msg << "Unable to parse line " << line_count << " of header file at " << header_str << std::endl;
                CUDART_ERROR(err_msg);
            } else {
                MeshBlockInfo mb_info;
                mb_info.mb_size = mb_size;
                mb_info.xl = vec3(xl,yl,zl);
                mb_info.xr = vec3(xr,yr,zr);
                mb_info.mb_dims = vec3(nx,ny,nz);
                // flexload data
                mb_info.mb_origin = 0.5 * (mb_info.xl + mb_info.xr);
                mb_info.mb_radius = (mb_info.xl - mb_info.mb_origin).vector_mag();
                all_mb_info.push_back(mb_info);
                npy_floats += mb_size;
            }
            line_count++;
        }
    } else {
        std::stringstream err_msg;
        err_msg << "### FATAL ERROR in meshblock.hpp ####\n";
        err_msg << "Unable to open header file at " << header_str << std::endl;
        CUDART_ERROR(err_msg);
    } // end header read

    if (verbose) { 
        std::chrono::duration<float> header_init_dur = (std::chrono::steady_clock::now() - header_init_start);
        printf("parsed header             (device)            %.6fs\n",header_init_dur.count());
    }

    // allocate space on host
    if (host_malloc) {
        h_bytes = npy_floats * sizeof(float);
        auto h_alloc_start = std::chrono::steady_clock::now();
        h_all_data = (float*) malloc(h_bytes);
        if (verbose) { 
            std::chrono::duration<float> h_alloc_dur = (std::chrono::steady_clock::now() - h_alloc_start);
            printf("malloc data               (host)              %.6fs\n",h_alloc_dur.count());
        }
    }
    
    // load mb data into host memory
    auto npy_read_start = std::chrono::steady_clock::now();
    int mem_offset = 0;
    for (int n = 0; n < all_mb_info.size(); n++) {
        // load meshblock data as (nx,ny,nz,p) where p = 1 or 4
        std::string npy_str = input_str + "/meshblock" + zero_pad_str(n, num_zero_pad) + ".npy";
        bool file_exists = std::filesystem::is_regular_file(npy_str);
        if (!file_exists) {
            std::stringstream err_msg;
            err_msg << "### FATAL ERROR in meshblock.hpp ###\n";
            err_msg << "Unable to locate input file at " << npy_str << std::endl;
            CUDART_ERROR(err_msg);
        }

        std::vector<unsigned long int> data_shape = npy_to_host(npy_str, h_all_data, h_bytes, verbose, false);
        bool beta_in_data = (data_shape.size() > 3);
        all_mb_info[n].beta_in_data = beta_in_data;
        all_mb_info[n].mem_start = mem_offset;
        all_mb_info[n].mb_index = n;
        
        // check dimensions
        if ((data_shape.size() == 3) && (relativistic)) {
            std::stringstream err_msg;
            err_msg << "### FATAL ERROR in meshblock.hpp ###\n";
            err_msg << "Relativistic flagged, but did not find velocity data in " << npy_str << std::endl;
            CUDART_ERROR(err_msg);
        }        
        // offset start of next meshblock by size of current meshblock
        mem_offset += data_shape.size();
    } // end mb loop

    if (verbose) {
        std::chrono::duration<float> npy_read_dur = (std::chrono::steady_clock::now() - npy_read_start);
        printf("npy read/memcpy           (host)              %.6fs\n",npy_read_dur.count());
    }

    return all_mb_info;
}

__device__ float MeshBlock::calc_trace(const Ray &r, TraceArgs trace_args) {
    // calculate the weighted path of a given ray through the MeshBlock
    float s_entry, s_exit, trace = 0;
    bool hit = calc_mb_intercept(r, s_entry, s_exit);
    if (hit) { // valid intercept found
        // prep arrays for orientation
        int cell[3] = {0, 0, 0}; // convert to vec3? typesafe?
        float ds[3] = {0.0, 0.0, 0.0};
        float s_next_intercept[3] = {0.0, 0.0, 0.0};
        int exit_cond[3] = {0, 0, 0};
        int step_dir[3] = {0, 0, 0};
        
        // if using lookback, automatically apply fast-forward along ray path
        if (trace_args.lookback) {
            float t_min = trace_args.snapshot_dt * (trace_args.snapshot_index - 1); // earliest contributing field
            float t_max = trace_args.snapshot_dt * (trace_args.snapshot_index + 1); // latest contributing field
            float s_min = trace_args.c * (trace_args.t_obs - t_max);                // shallowest contributing field
            float s_max = trace_args.c * (trace_args.t_obs - t_min);                // deepest contributing field
            if (s_max < s_entry) { // meshblock intersection too deep for observer time
                return 0.0; 
            } else if (s_min > s_exit) { // meshblock interseciton too shallow for observer time
                return 0.0;
            }
            s_entry = (s_min > s_entry) ? s_min : s_entry;
            s_exit = (s_max < s_exit) ? s_max : s_exit;
        } // end fast-forward

        // prepare for traversal
        vec3 mb_entrance = r.march(s_entry);
        for (int i = 0; i <= 2; i++) {
            // identify entry cell
            float ray_mb_orgin = mb_entrance[i] - xl[i];
            cell[i] = int_clamp(ray_mb_orgin / dx[i], 0, (int)mb_dims[i] - 1);
            // identify trace orientation
            if (r.sign[i]) { 
                step_dir[i] = -1; // traverse backwards
                exit_cond[i] = -1; // stop walk when leading edge reached
                ds[i] = - dx[i] * r.inv_normal[i];
                s_next_intercept[i] = s_entry + (cell[i] * dx[i] - ray_mb_orgin) * r.inv_normal[i];
            } else {
                step_dir[i] = 1; // traverse forwards
                exit_cond[i] = (int)mb_dims[i]; // stop walk when tailing edge reached
                ds[i] = dx[i] * r.inv_normal[i];
                s_next_intercept[i] = s_entry + ((cell[i]+1) * dx[i] - ray_mb_orgin) * r.inv_normal[i];
            } // end if sign
        } // end for direction

        // perform traversal
        float s_current = s_entry;
        while (s_current < s_exit) { // terminate on mb exit
            // identify next step direction
            int k = (((s_next_intercept[0] < s_next_intercept[1]) << 2) +
                    ((s_next_intercept[0] < s_next_intercept[2]) << 1) +
                    ((s_next_intercept[1] < s_next_intercept[2])));
            int axis = axes_bitmap[k];

            // determine dwell
            float dwell = s_next_intercept[axis] - s_current; 

            // add local cell to trace 
            int cell_index = cell[0] * (int)mb_dims[1] * (int)mb_dims[2]
                            + cell[1] * (int)mb_dims[2] + cell[2];
            int data_index = mem_start;
            if (beta_in_data) {
                data_index += cell_index * 4;
            } else {
                data_index += cell_index;
            }
            
            float trace_weight = dwell;
            if (trace_args.relativistic) {
                vec3 beta_vec(all_data[data_index+1],
                                all_data[data_index+2],
                                all_data[data_index+3]);
                trace_weight *= calc_boost_factor(beta_vec, r.normal, trace_args.doppler_index);
            }
            if (trace_args.lookback) {
                // apply temporal lerp and edge handling
                float midpoint_s = s_current + 0.5 * dwell;
                trace_weight *= calc_lookback_factor(midpoint_s, trace_args);
            }
            trace += trace_weight * all_data[data_index]; 
            trace = fmaxf(trace,0.0f);
            
            // update position of ray head
            s_current = s_next_intercept[axis]; // += dwell
            cell[axis] += step_dir[axis];
            s_next_intercept[axis] += ds[axis];

            // check for termination (necessary?)
            if (cell[axis] == exit_cond[axis]) break;
        } // end while traversall
    } // end if hit
    return trace;
}

__global__ void init_meshblock(MeshBlockInfo mb_info, MeshBlock **mb_list, float *data) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        mb_list[mb_info.mb_index] = new MeshBlock(mb_info, data);
    }
    return;
}

__global__ void free_meshblock(MeshBlock **mb) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        delete *mb;
    }
}

__global__ void print_meshblock_properties(MeshBlock **mb) {
    int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thr_idx == 0) {
        printf("printing data from mb...\n");
        printf("xl = (%.3f, %.3f, %.3f)\n", (*mb)->xl[0], (*mb)->xl[1], (*mb)->xl[2]);
        printf("xr = (%.3f, %.3f, %.3f)\n", (*mb)->xr[0], (*mb)->xr[1], (*mb)->xr[2]);
        printf("data_size = %d\n", (*mb)->mb_size);
    }  
}

__device__ int MeshBlock::int_clamp(float f, float l, float r) {
    return max(l, min(std::floor(f), r));
}

__device__ MeshBlock::MeshBlock(MeshBlockInfo mb_info, float *data) {
    xl = mb_info.xl;
    xr = mb_info.xr;
    all_data = data;
    mem_start = mb_info.mem_start;
    mb_dims = mb_info.mb_dims;
    mb_size = mb_info.mb_size;
    dx = (xr - xl) / mb_dims;
    beta_in_data = mb_info.beta_in_data;
}

__device__ bool MeshBlock::calc_mb_intercept(const Ray &r, float &s_entry, float &s_exit) {
    s_entry = 0.0, s_exit = 0.0;
    float temp_s_min, temp_s_max, s_min, s_max;
    // iterate over coordinate axes
    for (int i = 0; i <= 2; i++) {
        // find intercept with front/back faces
        temp_s_min = (get_edge(r.sign[i])[i] - r.origin[i]) * r.inv_normal[i];
        temp_s_max = (get_edge(1 - r.sign[i])[i] - r.origin[i]) * r.inv_normal[i];
        if (i == 0) {
            s_min = temp_s_min;
            s_max = temp_s_max;
            continue;
        }

        // if intercept after current back face, no collision
        if ((s_min > temp_s_max) or (temp_s_min > s_max)) {
            return false;
        }

        // store last entry, and first exit parameters
        s_min = (temp_s_min > s_min) ? temp_s_min : s_min;
        s_max = (temp_s_max < s_max) ? temp_s_max : s_max;
    }
    s_entry = s_min;
    s_exit = s_max;
    return true;
}

#endif