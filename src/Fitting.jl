using LsqFit, DataFrame, Statistics, StatsBase
function fit_and_calculate(data::DataFrame, 
    x_col::Union{Symbol,String}, y_col::Union{Symbol,String}, 
    fit_func::Function,init_param::VecOrMat,
    r_dig::Integer = 5)::NamedTuple

    fit = curve_fit(fit_func, data[!,x_col], data[!,y_col], init_param)
    rounded_params = round.(fit.param, digits = r_dig)
    param_error = standard_errors(fit)
    rounded_param_error = round.(param_error, digits = r_dig)
    param_high = fit.param .+ param_error
    param_low = fit.param .- param_error
    fitted_data = fit_func(data[!,x_col],fit.param)
    fitted_data_lower = fit_func(data[!,x_col],param_low)
    fitted_data_upper = fit_func(data[!,x_col],param_high)
    low_ribb = fitted_data .- fitted_data_lower
    high_ribb = fitted_data_upper .- fitted_data

    return (params = fit.param, r_param = rounded_params,
    r_err = rounded_param_error, prediction = fitted_data,
    ribb = (low_ribb, high_ribb), model = fit)
end
function fit_and_calculate(data::DataFrame, 
    x_col::Union{Symbol,String}, y_col::Union{Symbol,String}, 
    fit_func::Function,init_param::VecOrMat,
    data_weight::VecOrMat,r_dig::Integer = 5)::NamedTuple

    fit = curve_fit(fit_func, data[!,x_col], data[!,y_col],
    data_weight, init_param)
    rounded_params = round.(fit.param, digits = r_dig)
    param_error = standard_errors(fit)
    rounded_param_error = round.(param_error, digits = r_dig)
    param_high = fit.param .+ param_error
    param_low = fit.param .- param_error
    fitted_data = fit_func(data[!,x_col],fit.param)
    fitted_data_lower = fit_func(data[!,x_col],param_low)
    fitted_data_upper = fit_func(data[!,x_col],param_high)
    low_ribb = fitted_data .- fitted_data_lower
    high_ribb = fitted_data_upper .- fitted_data

    return (params = fit.param, r_param = rounded_params,
    r_err = rounded_param_error, prediction = fitted_data,
    ribb = (low_ribb, high_ribb), model = fit)
end