#=
    This script reads a Diplodocus.jl simulation output file, builds an uniform Cartesian grid over the simulation domain, constructs a statically measured emissivity value per cell based on the user supplied interactions.
    
    The emissivities are then output as a .npy file to be rendered using cuDART.

    The viewing angle must be supplied. 

=#

using DiplodocusTransport
using StaticArrays
using LinearAlgebra
using ProgressMeter

fileName = "M87_t256_8x8_med_a5.jld2"

fileLocation = joinpath(pwd(),"..","..","Papers","Thesis","Thesis_julia","M87Sims");

(PhaseSpace,sol) = SolutionFileLoad(fileLocation,fileName);
CHAR_l = PhaseSpace.Characteristic.CHAR_length

# domain size and Cartesian grid 
resolution = 1;

# set camera angle and position  
theta = 0.0 * pi / 180
phi = 0.0
camera_distance = 16.8e6 * DiplodocusTransport.CONST_pcs / CHAR_l # distance from observer to origin in code units (16.8 Mpc for M87)
camera_frequency = 1e11 # 10^11 Hz (roughly EHT)

"""
    DiplodocusDomainToCartesianGrid(PhaseSpace,resolution)

Convert Diplodocus Domain into a Cartesian uniform grid of `resolution` cells per code unit length. This is used to build a grid for cuDART rendering of simulation outputs.
"""
function GeneratecuDARTCartesianGrid(PhaseSpace::PhaseSpaceStruct,XDom::Tuple{Float64,Float64},YDom::Tuple{Float64,Float64},ZDom::Tuple{Float64,Float64},NX::Int64,NY::Int64,NZ::Int64)

    coordinates = PhaseSpace.Spacetime.coordinates
    x_r = PhaseSpace.Grids.xr 
    y_r = PhaseSpace.Grids.yr 
    z_r = PhaseSpace.Grids.zr

    Xr = collect(range(XDom[1],XDom[2],NX+1))
    Yr = collect(range(YDom[1],YDom[2],NY+1))
    Zr = collect(range(ZDom[1],ZDom[2],NZ+1))

    XGrid = DiplodocusTransport.meanVector(Xr)
    YGrid = DiplodocusTransport.meanVector(Yr)
    ZGrid = DiplodocusTransport.meanVector(Zr)

    Grid = fill((0.0,0.0,0.0),NX,NY,NZ)
    inGrid = Int64[]

    for (X_idx,X) in enumerate(XGrid), (Y_idx,Y) in enumerate(YGrid), (Z_idx,Z) in enumerate(ZGrid)
        Grid[X_idx,Y_idx,Z_idx] = (X,Y,Z) 
        if coordinates isa Paraboloidal
            ρ = sqrt(X^2 + Y^2)
            x = mod(atan(Y,X),2pi)
            y = sqrt(sqrt(ρ^2+Z^2)+Z)
            z = sqrt(sqrt(ρ^2+Z^2)-Z)
            if y <= y_r[end] && z <= z_r[end] 
                append!(inGrid,(Z_idx-1)*NY*NX+(Y_idx-1)*NX+X_idx)
            end
        end
    end

    return Grid, inGrid

end

XDom = (-225.0,225.0)
YDom = (-225.0,225.0)
ZDom = (-112.5,112.5)
NX = 50
NY = 50
NZ = 25
Grid, inGrid = GeneratecuDARTCartesianGrid(PhaseSpace,XDom,YDom,ZDom,NX,NY,NZ);

# setup for emission and binary interactions
Precision = Float32 
Backend = DiplodocusTransport.CPUBackend()

Binary_Domain = InclusiveDomainMask(PhaseSpace)
binaries=BinaryInteraction[BinaryInteraction("Pho","Pho","Ele","Pos"),BinaryInteraction("Ele","Pho","Ele","Pho")]

Sync_Domain = InclusiveDomainMask(PhaseSpace)
B_sampled::Vector{Float64} = [1e0,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7]
emissions=EmissiveInteraction[EmissiveInteraction(name1="Ele",name2="Ele",name3="Pho",EmiName="Sync",Ext_sampled=B_sampled,mode=Ani(),Force=true,Domain=Sync_Domain)]

DataDirectory = joinpath(pwd(),"..","DiplodocusCollisions","Data","M87");

Bin = BuildBinaryMatrices(PhaseSpace,binaries,Binary_Domain,DataDirectory;Bin_corrected=true,Bin_sparse=true,symmetric=true)
Emi = BuildEmissionMatrices(PhaseSpace,emissions,DataDirectory,Bin;Emi_corrected=true)
Flux = BuildFluxMatrices(PhaseSpace,AbstractForce[]) # needed for volume element and A flux

function GenerateChangeInDistribution(PhaseSpace::PhaseSpaceStruct,sol::OutputStruct,Bin::DiplodocusTransport.BinaryMatricesStruct,Emi::DiplodocusTransport.EmissionMatricesStruct,Flux::DiplodocusTransport.FluxMatricesStruct)

    Momentum = PhaseSpace.Momentum
    Spacetime = PhaseSpace.Spacetime
    Grids = PhaseSpace.Grids
    dt0 = Spacetime.dt0
    x_num = Spacetime.x_num
    y_num = Spacetime.y_num
    z_num = Spacetime.z_num
    x_r = PhaseSpace.Grids.xr 
    y_r = PhaseSpace.Grids.yr 
    z_r = PhaseSpace.Grids.zr
    n_space = x_num * y_num * z_num
    name_list = PhaseSpace.name_list
    p_num_list = Momentum.px_num_list
    u_num_list = Momentum.py_num_list
    h_num_list = Momentum.pz_num_list
    n_momentum = sum(p_num_list.*u_num_list.*h_num_list) 
    species_index = findfirst(x->x=="Pho",PhaseSpace.name_list)
    p_num = p_num_list[species_index]
    u_num = u_num_list[species_index]
    h_num = h_num_list[species_index]
    photon_offset = Grids.momentum_species_offset[species_index]
    photon_start = photon_offset + 1
    photon_end = photon_offset + p_num*u_num*h_num
    n_momentum_photon = p_num*u_num*h_num

    M_Bin_Mul_Step = zeros(n_momentum,n_momentum)
    M_Bin_Mul_Step_photon = @view(M_Bin_Mul_Step[photon_start:photon_end,:])
    M_Bin_Mul_Step_reshape = reshape(M_Bin_Mul_Step,n_momentum^2)

    dfdt_space = Vector{Vector{Float64}}(undef,n_space) # a vector of photon df/dt for each spatial cell in the simulation domain

    df_tmp = zeros(Float64,n_momentum_photon)
    df_photon = zeros(Float64,n_momentum_photon)
    dfdt_photon = zeros(Float64,n_momentum_photon)

    f = copy(sol.f[end])

    # evaluate df/dt for each spatial cell in the simulation domain 
    @showprogress for off_space in 0:n_space-1 

        start_idx = n_momentum*off_space+1
        end_idx = n_momentum*(off_space+1)
        Vol = Flux.Vol[off_space+1]
        # calculate df/dt in sim frame 
        # Adf = MBin*f*f*V + MEmi*f   # V contains dt0 and MEmi contains V
        # df/dt0 = A^{-1}(MBin*f*f*V + MEmi*f)/dt0
        f_local = @view(f[start_idx:end_idx])
        # MEmi part 
        MEmi_photon = @view(Emi.M_Emi[off_space+1][photon_start:photon_end,:])
        if !isempty(Emi.Emission_list) && !isnothing(Emi.M_Emi[off_space+1])
            mul!(df_tmp,MEmi_photon,f_local)
            @. df_photon += df_tmp 
        end
        # MBin part 
        MBin = Bin.M_Bin
        if !isempty(Bin.Binary_list)
            mul!(M_Bin_Mul_Step_reshape,MBin,f_local,Vol,0.0)
            mul!(df_tmp,M_Bin_Mul_Step_photon,f_local)
            @. df_photon += df_tmp
        end
        # invA and dt0 parts
        @. dfdt_photon = df_photon / @view(Flux.Ap_Flux[start_idx+photon_offset:start_idx+photon_offset+n_momentum_photon-1]) / dt0 # TODO: correct normalisation due to definition of f

        dfdt_space[off_space+1] = copy(dfdt_photon)
    end

    return dfdt_space

end

dfdt_space = GenerateChangeInDistribution(PhaseSpace,sol,Bin,Emi,Flux)

function GenerateEmissivity(PhaseSpace::PhaseSpaceStruct,Grid::Array{Tuple{Float64,Float64,Float64},3},inGrid::Vector{Int64},dfdt_space::Vector{Vector{Float64}},camera_distance::Float64,camera_frequency::Float64,theta::Float64)

    Momentum = PhaseSpace.Momentum
    Spacetime = PhaseSpace.Spacetime
    x_num = Spacetime.x_num
    y_num = Spacetime.y_num
    z_num = Spacetime.z_num
    x_r = PhaseSpace.Grids.xr 
    y_r = PhaseSpace.Grids.yr 
    z_r = PhaseSpace.Grids.zr
    p_num_list = Momentum.px_num_list
    u_num_list = Momentum.py_num_list
    h_num_list = Momentum.pz_num_list
    species_index = findfirst(x->x=="Pho",PhaseSpace.name_list)
    p_num = p_num_list[species_index]
    u_num = u_num_list[species_index]
    h_num = h_num_list[species_index]
    coordinates = PhaseSpace.Spacetime.coordinates

    vec_origin_to_camera = zeros(Float64,3)
    vec_origin_to_camera[1] = camera_distance*sin(theta)
    vec_origin_to_camera[2] = 0.0
    vec_origin_to_camera[3] = camera_distance*cos(theta)

    vec_point_to_camera = zeros(Float64,3)
    vec_origin_to_point = zeros(Float64,3)

    pvec_stationary_observer = zeros(Float64,4)
    pvec_tmp = zeros(Float64,4)
    pvec_tmp2 = zeros(Float64,4)
    pvec_simulation = zeros(Float64,4)
    point_simcoordinates = @MVector zeros(Float64,4)
    point_cartesian = @MVector zeros(Float64,4)

    metric = PhaseSpace.Spacetime.metric
    tetrad = PhaseSpace.Spacetime.tetrad

    inve_paraboloidal_to_simulation = @MMatrix zeros(Float64,4,4)  # transpose(inve) converts paraboloidal coordinate basis to simulation observer basis
    e_simulation_to_paraboloidal = @MMatrix zeros(Float64,4,4) # transpose(e) converts local observer basis to paraboloidal coordinate basis
    inve_cartesian_to_stationary = @MMatrix zeros(Float64,4,4) # transpose(inve) converts cartesian coordinate vector to stationary observer vector
    e_stationary_to_cartesian = @MMatrix zeros(Float64,4,4) # transpose(e) converts stationary observer vector to cartesian coordinate vector

    g = @MMatrix zeros(Float64,4,4)

    freq_to_momentum = DiplodocusTransport.CONST_ħ * 2pi / (DiplodocusTransport.CONST_mEle * DiplodocusTransport.CONST_c^2)

    emissivity::Array{Float32,3} = zeros(Float32,NX,NY,NZ)

    TetradComponents!(point_cartesian, e_simulation_to_paraboloidal, metric, Paraboloidal(), tetrad)

    @showprogress for idx in inGrid

        X,Y,Z = Grid[idx]
  
        # find u,v,ϕ coordinate of X,Y,Z
        ρ = sqrt(X^2 + Y^2)
        x = mod(atan(Y,X),2pi)
        y = sqrt(sqrt(ρ^2+Z^2)+Z)
        z = sqrt(sqrt(ρ^2+Z^2)-Z)
        #println("x: $x, y: $y, z: $z, ρ: $ρ")
        # find u_idx, v_idx, ϕ_idx
        x_idx = length(x_r) == 2 ? 1 : searchsortedfirst(x_r, x)-1
        y_idx = searchsortedfirst(y_r, y)-1
        z_idx = searchsortedfirst(z_r, z)-1
        # calculate off_space to get dfdt for this simulation spatial cell
        off_space = (x_idx-1)*y_num*z_num+(y_idx-1)*z_num+z_idx-1
        dfdt = dfdt_space[off_space+1]
        # define vector to camera 
        vec_origin_to_point[1] = X
        vec_origin_to_point[2] = Y
        vec_origin_to_point[3] = Z
        @. vec_point_to_camera = vec_origin_to_camera - vec_origin_to_point
        # work out local momentum direction of camera
        vec_point_to_camera_norm = norm(vec_point_to_camera) 
        pvec_stationary_observer[1] = 1.0
        pvec_stationary_observer[2] = vec_point_to_camera[1] / vec_point_to_camera_norm
        pvec_stationary_observer[3] = vec_point_to_camera[2] / vec_point_to_camera_norm
        pvec_stationary_observer[4] = vec_point_to_camera[3] / vec_point_to_camera_norm
        pvec_stationary_observer .*= camera_frequency * freq_to_momentum
        #println("pvec_stationary_observer: $pvec_stationary_observer")
        #MetricComponents!(point_cartesian,g,metric,Cartesian())
        #println(pvec_stationary_observer' * g * pvec_stationary_observer)
        # generate tetrads 
        point_simcoordinates[1] = 0.0
        point_simcoordinates[2] = x 
        point_simcoordinates[3] = y 
        point_simcoordinates[4] = z
        CoordinateTransform!(point_simcoordinates,point_cartesian,coordinates,Cartesian()) # this consistently converts coordinates avoiding a float error
        #println("$point_cartesian")
        #println("$point_simcoordinates")
        InverseTetradComponents!(point_cartesian,inve_cartesian_to_stationary,metric,Cartesian(),StationaryObserverTetrad()) 
        InverseTetradComponents!(point_simcoordinates,inve_paraboloidal_to_simulation,metric,Paraboloidal(),tetrad) 
        TetradComponents!(point_simcoordinates,e_simulation_to_paraboloidal,metric,Paraboloidal(),tetrad)  
        TetradComponents!(point_cartesian,e_stationary_to_cartesian,metric,Cartesian(),StationaryObserverTetrad()) 
        # convert stationary observer momentum vector to simulation measured momentum vector
        mul!(pvec_tmp2,e_stationary_to_cartesian',pvec_stationary_observer) # converts momentum vector measured in stationary observer frame in Cartesian coordinates to momentum vector measured in stationary observer frame in stationary observer basis
        CoordinateBasisTransform!(pvec_tmp2,pvec_tmp,point_cartesian,Cartesian(),Paraboloidal()) # converts momentum vector measured in stationary observer frame in Cartesian coordinates to momentum vector measured in the paraboloidal basis in paraboloidal coordinates
        #println("pvec: $(e_stationary_to_cartesian' * pvec_stationary_observer)")
        #println("pvec: $(pvec_tmp)")
        # Check still null vector 
        #MetricComponents!(point_simcoordinates,g,metric,Paraboloidal())
        #println(pvec_tmp' * g * pvec_tmp)
        mul!(pvec_simulation, inve_paraboloidal_to_simulation', pvec_tmp) # convert momentum vector measured in the paraboloidal basis in paraboloidal coordinates to local observer basis in paraboloidal coordinates
        #println("pvec_simulation: $pvec_simulation")
        # find location of simulation vector in momentum space grid
        p_sim = pvec_simulation[1] # momentum p in sim frame
        u_sim = pvec_simulation[4]/pvec_simulation[1] # momentum u=cos(θ) = pcos(θ)/p in sim frame
        h_sim = mod(atan(pvec_simulation[3],pvec_simulation[2]),2pi) # momentum ϕ in sim frame tan(ϕ) = psin(θ)sin(ϕ)/psin(θ)cos(ϕ) in sim frame
        if p_sim <= 0.0
            println("idx: $idx, $X, $Y, $Z, p_sim: $p_sim")
        end
        p_loc = DiplodocusTransport.location(PhaseSpace.Momentum.px_low_list[2],PhaseSpace.Momentum.px_up_list[2],PhaseSpace.Momentum.px_num_list[2],p_sim,PhaseSpace.Momentum.px_grid_list[2])
        u_loc = DiplodocusTransport.location(-1.0,1.0,PhaseSpace.Momentum.py_num_list[2],u_sim,PhaseSpace.Momentum.py_grid_list[2])
        h_loc = DiplodocusTransport.location(0.0,2*pi,PhaseSpace.Momentum.pz_num_list[2],h_sim,PhaseSpace.Momentum.pz_grid_list[2])
        # work out element of dfdt for this momentum location
        f_idx = (h_loc-1)*u_num*p_num + (u_loc-1)*p_num+p_loc
        # work out dfdt_camera 
        dfdt_camera = dfdt[f_idx]
        # add to array of dfdt_camera values
        emissivity[idx] = Float32(dfdt_camera)

    end

    return emissivity

end

emissivity = GenerateEmissivity(PhaseSpace,Grid,inGrid,dfdt_space,camera_distance,camera_frequency,theta);

n_momentum = 4374
    M_Bin_Mul_Step = zeros(Float32,n_momentum,n_momentum)
    M_Bin_Mul_Step_reshape = reshape(M_Bin_Mul_Step,n_momentum^2)
f_local = @view(sol.f[end][1:n_momentum])
MBin = Bin.M_Bin
@btime mul!($M_Bin_Mul_Step_reshape,$MBin,$f_local)
mul!(df_tmp,M_Bin_Mul_Step,f_local)

# convert emissivity to .npy array
using PythonCall
using CondaPkg
CondaPkg.add("numpy")
np = pyimport("numpy")
np.save("emissivity_x_wBinary.npy",np.array(emissivity,order="C"))

A = rand(200000,5000)
B = rand(5000)
C = rand(200000)
@btime mul!($C,$A,$B)

using SparseArrays
using MKLSparse
As = sprand(2000000,50000,0.01)
Bs = rand(50000)
Cs = rand(2000000)
@btime mul!($Cs,$As,$Bs)