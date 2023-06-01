using CSV, DataFrames
using Plots, StatsPlots
using StableRNGs
using ProgressMeter
using Dates
using Statistics
using DataInterpolations

include("convolutions_1d.jl")


raw_paths = Dict("central_node_1" => "./data/raw/central_node_1",
                 "central_node_2" => "./data/raw/central_node_2",
                 "central_node_4" => "./data/raw/central_node_4"
                 )

for (key,val) ∈ raw_paths
    println(key, "\t", isdir(val))
    @assert ispath(val)
end


# load a test path for a specific year/month/day to try reading the files from
raw_paths["central_node_4"]

basepath = joinpath(raw_paths["central_node_4"], "2023/05/12")
@assert isdir(basepath)


test_paths = []
for (root, dirs, files) ∈ walkdir(basepath)
    for file ∈ files
        println(joinpath(root, file))
        push!(test_paths, joinpath(root, file))
    end
end


# create a function to read the datetime string
# we need to deal with the problem of
# nanosecond precision
function get_datetime(datestring::String)
    f_string = dateformat"yyyy-mm-dd HH:MM:SS.sss"

    split_string = split(datestring, ".")
    datetime = join([split_string[1], split_string[2][1:3]], ".")  # only keep milliseconds
    return DateTime(datetime, f_string)
end

function get_datetime(datestring::AbstractString)
    get_datetime(String(datestring))
end

sensors_to_include = [
    "IPS7100",
#    "APDS9002",
    "BME680",
#    "GUV001",
#    "TB108L",
#    "TMG3993",
#    "TSL2591",
#    "VEML6075"
]


function get_df(path)
    df = CSV.File(path) |> DataFrame
    @assert "dateTime" ∈ names(df)
    df.dateTime .= get_datetime.(df.dateTime)
    return df
end

function add_milliseconds_column!(df::DataFrame, T_start::DateTime)
    df.ΔT = [(t .- T_start).value for t ∈ df.dateTime]
end



function process_one_day(datapath::String, sensors_to_include::Vector{String}; Δt_out=15)
    # load all dataframes from the day
    dfs = DataFrame[]

    data_paths = []
    # generate
    for (root, dir, files) ∈ walkdir(datapath)
        for file ∈ files
            if any(occursin.(sensors_to_include, file))
                push!(data_paths, joinpath(root, file))
            end
        end
    end

    @assert length(data_paths) == length(sensors_to_include)

    for path ∈ data_paths
        if any(occursin.(sensors_to_include, path))
            push!(dfs, get_df(path))
        end
    end

    # compute maximum start time
    T_start = maximum([df.dateTime[1] for df ∈ dfs])
    T_end = minimum([df.dateTime[end] for df ∈ dfs])

    # compute Δt column for time since T_start in milliseconds
    for df ∈ dfs
        add_milliseconds_column!(df, T_start)
    end


    Δts = []
    for df ∈ dfs
        Δt = mean(df.ΔT[2:end] .- df.ΔT[1:end-1])
        push!(Δts, Δt)
    end

    #Δt_out = 15
    ts = 0:(Δt_out * 1000):(T_end - T_start).value

    # create output dataframe to store results
    df_out = DataFrame()
    df_out.ts = ts
    df_out.dateTime = T_start .+ Millisecond.(ts)

    # loop over dataframes and add interpolated columns to output
    for df ∈ dfs
        ts_in = df.ΔT

        for colname ∈ names(df[:, Not([:dateTime, :ΔT])])
            itp = CubicSpline(df[:,colname], ts_in)

            df_out[!, colname] = itp.(ts)
        end
    end

    return df_out
end


# test out the code

df = process_one_day(basepath, sensors_to_include)
test_paths[1]

basepath2 = joinpath(raw_paths["central_node_4"], "2023/05/11")
df2 = process_one_day(basepath2, sensors_to_include)


l_kernel = 10  # 5 minute windowed average
use_gauss = false
k = mean_kernel(l_kernel)
if use_gauss
    k = gaussian_kernel_1D(l_kernel)
end


function smooth_data(df, k)
    df_new = copy(df)
    cols_to_use = [n for n ∈ names(df) if !(n ∈ ["ts", "dateTime", "device_name"])]
    for col_name ∈ cols_to_use
        df_new[:,col_name] .= convolve(df[:,col_name], k)
    end

    return df_new
end

# test out the smoothing on pm 2.5
df_smooth = smooth_data(df2, k)
p1 = plot(df2.pm2_5[1:200], label="raw", lw=2)
plot!(p1, df_smooth.pm2_5[1:200], label="smoothed", lw=2)





# the IPS7100 sensor becomes available on 2021/08/18

isdir.(readdir(raw_paths["central_node_4"]))

basepath_2021 = joinpath(raw_paths["central_node_4"], "2021")
months = filter(isdir,joinpath.(basepath_2021, readdir(basepath_2021)))
months = [m for m ∈ months if parse(Int,(split(m, "/")[end])) > 8]
days2021 = []
for month ∈ months
    push!(days2021, filter(isdir,joinpath.(month, readdir(month))))
end
days2021 = vcat(days2021...)

basepath_2022 = joinpath(raw_paths["central_node_4"], "2022")
months = filter(isdir,joinpath.(basepath_2022, readdir(basepath_2022)))
days2022 = []
for month ∈ months
    push!(days2022, filter(isdir,joinpath.(month, readdir(month))))
end
days2022 = vcat(days2022...)

basepath_2023 = joinpath(raw_paths["central_node_4"], "2023")
months = filter(isdir,joinpath.(basepath_2023, readdir(basepath_2023)))
days2023 = []
for month ∈ months
    push!(days2023, filter(isdir,joinpath.(month, readdir(month))))
end
days2023 = vcat(days2023...)


paths_to_process = vcat(days2021..., days2022..., days2023...)

# process all of these files and put them in the processed folder
sensors_to_include
p = Progress(length(paths_to_process))
Threads.@threads for path ∈ paths_to_process
    outname = "measurements"*join(split(path, "/")[end-3:end], "_")*"-central_node_4.csv"

    try
        df = process_one_day(path, sensors_to_include)

        df = smooth_data(df, k)

        # add column with device name for book keeping
        df.device_name = ["Central Node 4" for _ ∈ 1:nrow(df)]

        # save to output folder
        CSV.write(joinpath("data/processed", outname), df)

    catch e
        println(e)
    end

    next!(p)
end
finish!(p)


# repeat for other nodes

basepath_2021 = joinpath(raw_paths["central_node_1"], "2021")
months = filter(isdir,joinpath.(basepath_2021, readdir(basepath_2021)))
months = [m for m ∈ months if parse(Int,(split(m, "/")[end])) > 8]
days2021 = []
for month ∈ months
    push!(days2021, filter(isdir,joinpath.(month, readdir(month))))
end
days2021 = vcat(days2021...)

basepath_2022 = joinpath(raw_paths["central_node_1"], "2022")
months = filter(isdir,joinpath.(basepath_2022, readdir(basepath_2022)))
days2022 = []
for month ∈ months
    push!(days2022, filter(isdir,joinpath.(month, readdir(month))))
end
days2022 = vcat(days2022...)

basepath_2023 = joinpath(raw_paths["central_node_1"], "2023")
months = filter(isdir,joinpath.(basepath_2023, readdir(basepath_2023)))
days2023 = []
for month ∈ months
    push!(days2023, filter(isdir,joinpath.(month, readdir(month))))
end
days2023 = vcat(days2023...)


paths_to_process = vcat(days2021..., days2022..., days2023...)

# process all of these files and put them in the processed folder
sensors_to_include

p = Progress(length(paths_to_process))
Threads.@threads for path ∈ paths_to_process
    outname = "measurements"*join(split(path, "/")[end-3:end], "_")*"-central_node_1.csv"

    try
        df = process_one_day(path, sensors_to_include)

        # add column with device name for book keeping
        df.device_name = ["Central Node 1" for _ ∈ 1:nrow(df)]

        # save to output folder
        CSV.write(joinpath("data/processed", outname), df)

    catch e
        println(e)
    end

    next!(p)
end
finish!(p)

# repeat for other nodes

basepath_2021 = joinpath(raw_paths["central_node_2"], "2021")
months = filter(isdir,joinpath.(basepath_2021, readdir(basepath_2021)))
months = [m for m ∈ months if parse(Int,(split(m, "/")[end])) > 8]
days2021 = []
for month ∈ months
    push!(days2021, filter(isdir,joinpath.(month, readdir(month))))
end
days2021 = vcat(days2021...)

basepath_2022 = joinpath(raw_paths["central_node_2"], "2022")
months = filter(isdir,joinpath.(basepath_2022, readdir(basepath_2022)))
days2022 = []
for month ∈ months
    push!(days2022, filter(isdir,joinpath.(month, readdir(month))))
end
days2022 = vcat(days2022...)

basepath_2023 = joinpath(raw_paths["central_node_2"], "2023")
months = filter(isdir,joinpath.(basepath_2023, readdir(basepath_2023)))
days2023 = []
for month ∈ months
    push!(days2023, filter(isdir,joinpath.(month, readdir(month))))
end
days2023 = vcat(days2023...)


paths_to_process = vcat(days2021..., days2022..., days2023...)

# process all of these files and put them in the processed folder
sensors_to_include

p = Progress(length(paths_to_process))
Threads.@threads for path ∈ paths_to_process
    outname = "measurements"*join(split(path, "/")[end-3:end], "_")*"-central_node_2.csv"

    try
        df = process_one_day(path, sensors_to_include)

        # add column with device name for book keeping
        df.device_name = ["Central Node 2" for _ ∈ 1:nrow(df)]

        # save to output folder
        CSV.write(joinpath("data/processed", outname), df)

    catch e
        println(e)
    end

    next!(p)
end
finish!(p)


# now for each dataframe, we want to let's create a new one with rows i and i-1 concatenated
# together to provide the model a way to intuit derivative information

basepath = "data/processed"
files = filter(x->endswith(x, ".csv"), joinpath.(basepath, readdir(basepath)))


function generate_multistep_df(path; Nwindow=2)
    df = CSV.File(path) |> DataFrame
    names(df)
    M_df = Matrix(df[:, Not([:ts, :dateTime, :device_name])])

    N = nrow(df)
    N_out = N-(Nwindow-1)

    M_out = hcat([M_df[i:(N_out+i-1),:] for i ∈ 1:Nwindow]...)

    col_names = names(df[:, Not([:ts, :dateTime, :device_name])])
    col_names_out = [col_names .* "_i"]
    for j ∈ 2:Nwindow
        push!(col_names_out, col_names .* "_i-$(j-1)")
    end
    col_names_out = vcat(col_names_out...)

    df_out = DataFrame(M_out, col_names_out)
    df_out.dateTime = df.dateTime[1:N_out]
    df_out.device_name = df.device_name[1:N_out]
    df_out.ts = df.ts[1:N_out]

    return df_out
end


df_final = DataFrame()
Nwindow = 2
@showprogress for f ∈ files
    df = generate_multistep_df(f; Nwindow=Nwindow)
    df_out = hcat(df[1:end-1,:], rename(x->x*"_next", df[2:end, Not([:ts, :dateTime, :device_name])]))
    append!(df_final, df_out)
#    push!(dfs, df[1:end-1,:])
#    push!(dfs_next, df[2:end, :])
end


dropmissing!(df_final) # drop any rows with missing values

# df_final, df_final_next = get_collected_dfs(files)


# save final output to new folder in data directory
if !isdir("data/combined")
    mkpath("data/combined")
end


gdf = groupby(df_final, :device_name)

keys(gdf)

# write each node to a separate file
CSV.write("data/combined/central-node-1_Nwindow-$(Nwindow).csv", gdf[(device_name="Central Node 1",)])
CSV.write("data/combined/central-node-2_Nwindow-$(Nwindow).csv", gdf[(device_name="Central Node 2",)])
CSV.write("data/combined/central-node-4_Nwindow-$(Nwindow).csv", gdf[(device_name="Central Node 4",)])

