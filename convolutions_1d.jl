
"""
    extend(v::AbstractVector, i)

Index into vector `v` at position `i`. If `i` is outside bounds of array, use the nearest values
"""
function extend(v::AbstractVector, i)
	  if i ∈ 1:length(v)
		    return v[i]
	  else
		    i_new = argmin(x->abs(x-i), [1, length(v)])
		    return v[i_new]
	  end
end



"""
    extend(M::AbstractMatrix, i, j)

Index into matrix at position `(i,j)`. If `(i,j)` is outside of
"""
function extend(M::AbstractMatrix, i, j)
	  if i ∈ axes(M,1) && j ∈ axes(M,2)
		    return M[i,j]
	  elseif i ∈ axes(M,1) && !(j ∈ axes(M,2))
		    j_new = argmin(x->abs(x-j), (1, size(M,2)))
		    return M[i, j_new]
	  elseif !(i ∈ axes(M,1)) && j ∈ axes(M,2)
		    i_new = argmin(x->abs(x-i), (1, size(M,1)))
		    return M[i_new, j]
	  else
		    i_new = argmin(x->abs(x-i), (1, size(M,1)))
		    j_new = argmin(x->abs(x-j), (1, size(M,2)))
		    return M[i_new, j_new]
	  end
end



"""
    convolve(v::AbstractVector, k)

Convolve the vector `v` with the kernel `k` according to

v′ᵢ = ∑ⱼ vᵢ₋ⱼ kⱼ

"""
function convolve(v::AbstractVector, k)
	  ℓ = (length(k)-1)÷2  # ÷ for integer division
	  return [sum([extend(v, i-m)*k[m+ℓ+1] for m ∈ -ℓ:ℓ]) for i ∈ 1:length(v)]
end


"""
    mean_kernel(l)

Construct a kernel of length `2l+1` which will perform a sliding mean when convolved with a time series `v`.
"""
function mean_kernel(l)
	  return ones(2l+1) ./ (2l +1)
end




function extend(M::AbstractMatrix, i, j, kₖ,kₗ)
	  return [extend(M, i-k, j-l) for k ∈ -kₖ:kₖ, l ∈ -kₗ:kₗ]
end


function convolve(M::AbstractMatrix, K::AbstractMatrix)
	  kₖ = (size(K,1)-1)÷2
	  kₗ = (size(K,2)-1)÷2

	  M_out = similar(M)
	  for j ∈ axes(M_out,2), i ∈ axes(M_out,1)
		    M_out[i,j] = sum(extend(M,i,j,kₖ,kₗ) .* K)
	  end
	  return M_out
end


gauss(x::Real; σ=1) = 1 / sqrt(2π*σ^2) * exp(-x^2 / (2 * σ^2))

function gaussian_kernel_1D(n; σ = 1)
	  res = gauss.(-n:n; σ=σ)
	  return res ./ sum(res)
end
