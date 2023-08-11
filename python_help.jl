### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 907ea08e-2d52-11ee-0218-67180aaaba6e
begin
	begin
		using Pkg
		Pkg.activate(".")
	end
end

# ╔═╡ 0273fe28-3cf4-4d92-96ed-b4b81b789e8c
using PyCall, Statistics, LinearAlgebra

# ╔═╡ c204a489-151d-4c37-8255-22803946a1c5
using .Iterators

# ╔═╡ e564e588-4336-4069-98bb-fb5cfa78a8f5
using LsqFit

# ╔═╡ 933ec515-4bbe-4208-a496-221edc71b232
begin
@pyinclude("python_calib.py")

chessboard_correspondences = py"getChessboardCorners"()   
end

# ╔═╡ bed0b3f8-83e6-4d7c-b54f-b9be0b5f71b9
image_cords, world_cords = chessboard_correspondences[1,:]

# ╔═╡ 6b057b4c-7d75-48bd-9f9d-96d827618755
begin

function get_normalization_matrix(pts, name="A")
	pts = Float64.(pts)
	x_mean, y_mean = mean(pts, dims=1)
	var_x, var_y = var(pts, dims=1;corrected=false)

	s_x , s_y = sqrt(2/var_x), sqrt(2/var_y)
	
	# println("Matrix: $(name) : meanx $(x_mean) , meany $(y_mean) , varx $(var_x) , vary $(var_y) , sx $(s_x) , sy $(s_y)")

	n = [s_x 0 -s_x*x_mean;0 s_y -s_y*y_mean; 0 0 1]
	# print(n)

	n_inv = [(1 ./ s_x) 0 x_mean; 0 (1 ./ s_y) y_mean;0 0 1]

	
	# @info "N:" n n_inv
	return Float64.(n), Float64.(n_inv)
end
	
function normalize_points(cords)
	views = size(cords)[1]

	ret_correspondences = [] 
    for i in 1:views
        imp, objp = chessboard_correspondences[i,:]
        N_x, N_x_inv = get_normalization_matrix(objp, "A")
        N_u, N_u_inv = get_normalization_matrix(imp, "B")
		val = ones(Float64,(54,1))
		
		normalized_hom_imp = hcat(imp, val)
        normalized_hom_objp = hcat(objp, val)

		for i in 1:size(normalized_hom_objp)[1]
			n_o = N_x * normalized_hom_objp[i,:]
            normalized_hom_objp[i,:] = n_o/n_o[end]

            n_u = N_u * normalized_hom_imp[i,:] 
            normalized_hom_imp[i,:]  = n_u/n_u[end]
		end

		normalized_objp =  normalized_hom_objp
		normalized_imp =  normalized_hom_imp
		push!(ret_correspondences, (imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv))
	end
	return ret_correspondences
end
end

# ╔═╡ 0e990a5f-c9a4-4326-8aec-5f5147507a6b
chessboard_correspondences_normalized = normalize_points(chessboard_correspondences)

# ╔═╡ edbd55cf-8f54-4184-a7f2-5c805c98bdee
function compute_view_based_homography(correspondence; reproj = 0)
    """
    correspondence = (imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv)
    """
	# @info correspondence
    image_points = correspondence[1]
    object_points = correspondence[2]
    normalized_image_points = correspondence[3]
    normalized_object_points = correspondence[4]
    N_u = correspondence[5]
    N_x = correspondence[6]
    N_u_inv = correspondence[7]
    N_x_inv = correspondence[8]

    N = size(image_points)[1]
 #    print("Number of points in current view : ", N)
	# print("\n")
	
    M = zeros(Float64, (2*N, 9))
    # println("Shape of Matrix M : ", size(M))

    # println("N_model\n", N_x)
    # println("N_observed\n", N_u)

	data = []
	for i in 1:54
		world_point, image_point = normalized_object_points[i,:], normalized_image_points[i,:]
		# @info "points:" world_point image_point
		u = image_point[1]
		v = image_point[2]
		X = world_point[1]
		Y = world_point[2]
		res = [-X -Y -1 0 0 0 X*u Y*u u;  0 0 0 -X -Y -1 X*v Y*v v]
		push!(data, res)
	end
	Amat = vcat(data...)
	# @info Amat
	u, s, vh = svd(Amat)
	# @show "Svd:" u s vh
	min_idx = findmin(s)[2]
	# @info vh min_idx
	# @info vh[:,min_idx]
	h_norm = vh[:,min_idx]
	# @info h_norm
	h_norm = reshape(h_norm,(3,3))'
	# @info h_norm
	
	h = (N_u_inv * h_norm) * N_x
    # @info "h:" N_u_inv h_norm
 #    # if abs(h[2, 2]) > 10e-8:
    h = h ./ h[3, 3]
    # print("Homography for View : \n", h )

    return h
end

# ╔═╡ bdff8656-8b51-4a33-912f-3b6128d757ed
begin
	H = []
	for correspondence in chessboard_correspondences_normalized
	    push!(H, compute_view_based_homography(correspondence; reproj=0))
	end
	H
end

# ╔═╡ e5ffc1b5-f5ef-45b1-8387-8610825151e5
begin
	function model(X, h)
		# @show X length(X)
		# @show h
		N = trunc(Int, length(X) / 2)
	    x_j = reshape(X, (2, N))'
		# @info "x_j:" x_j
	    projected = zeros(2*N)
		
	    for j in 1:N
	        x, y = x_j[j,:]
	        w = h[7]*x + h[8]*y + h[9]
	        projected[(2*j) - 1] = (h[1] * x + h[2] * y + h[3]) / w
	        projected[2*j] = (h[4] * x + h[5] * y + h[6]) / w
		end
		# @info "Projected:" projected length(projected)
		return projected
	end
	
	function jac_function(X, h)
		N = trunc(Int, length(X) /2)
		# @show N
	    x_j = reshape(X , (2, N))'
	    jacobian = zeros(Float64, (2*N, 9))
	    for j in 1:N
	        x, y = x_j[j,:]
	        sx = Float64(h[1]*x + h[2]*y + h[3])
	        sy = Float64(h[4]*x + h[5]*y + h[6])
	        w = Float64(h[7]*x + h[8]*y + h[9])
	        jacobian[(2*j) - 1,:] = [x/w, y/w, 1/w, 0, 0, 0, -sx*x/w^2, -sx*y/w^2, -sx/w^2]
	        jacobian[2*j,:] = [0, 0, 0, x/w, y/w, 1/w, -sy*x/w^2, -sy*y/w^2, -sy/w^2]
		end

		# @info "Jacobian:" jacobian length(jacobian)
	    return jacobian
	end
		
	function refine_homographies(H, correspondence; skip=false)
	    if skip
	        return H
		end
	
	    image_points = correspondence[1]
	    object_points = correspondence[2]
	    normalized_image_points = correspondence[3]
	    normalized_object_points = correspondence[4]
	    # N_u = correspondence[5]
	    N_x = correspondence[6]
	    N_u_inv = correspondence[7]
	    N_x_inv = correspondence[8]
		
		N = size(normalized_object_points)[1]
	    X = Float64.(collect(flatten(object_points')))
	    Y = Float64.(collect(flatten(image_points')))
	    h = collect(flatten(H'))
		# @show h
		# @show det(H)

		# @info "data:" X Y h

		fit = curve_fit(model, jac_function, Float64.(X), Float64.(Y), h;)

		if fit.converged
	        H =  reshape(fit.param,  (3, 3))
		end
		H = H/H[3, 3]
		
		return H
	end
end

# ╔═╡ 1f668b5b-22aa-43d8-9293-fcf9c2bed5fe
begin
	H_r = []

	for i in 1:length(H)
		# @info "Input Homography:" H[i]
	    h_opt = refine_homographies(H[i], chessboard_correspondences_normalized[i]; skip=false)
		# @info h_opt
	    # push!(H_r, h_opt')
		# @info "Refined Homography:" h_opt
		push!(H_r, h_opt')
	end     
	H_r
end

# ╔═╡ fe2c2081-67ab-4a6d-8d88-c81c42f3833a
function get_intrinsic_parameters(H_r)
    M = length(H_r)
    V = zeros(Float64, (2*M, 6))

    function v_pq(p, q, H)
        v = [
                H[1, p]*H[1, q] 
                (H[1, p]*H[2, q] + H[2, p]*H[1, q]) 
                H[2, p]*H[2, q] 
                (H[3, p]*H[1, q] + H[1, p]*H[3, q]) 
                (H[3, p]*H[2, q] + H[2, p]*H[3, q]) 
                H[3, p]*H[3, q]
            ]
        return v
	end 

    for i in 1:M
        H = H_r[i]
        V[(2*i)-1,:] = v_pq(1, 2, H)
        V[2*i,:] = v_pq(1,1, H) .- v_pq(2, 2, H)
	end 

    # solve V.b = 0
    u, s, vh = svd(V)
    # print(u, "\n", s, "\n", vh)
	# @info u
    b = vh[:,findmin(s)[2]]
	@info size(u) size(s) size(vh)
    print("V.b = 0 Solution : ", b)

    # according to zhangs method
    vc = (b[2]*b[4] - b[1]*b[5])/(b[1]*b[3] - b[2]^2)
    l = b[6] - (b[4]^2 + vc*(b[2]*b[3] - b[1]*b[5]))/b[1]
    alpha = sqrt((l/b[1]))
    beta = sqrt((l*b[1])/(b[1]*b[3] - b[2]^2))
    gamma = -1*((b[2])*(alpha^2) *(beta/l))
    uc = (gamma*vc/beta) - (b[4]*(alpha^2)/l)

    A = [ alpha gamma uc;
          0 beta vc;
            0 0 1.0;
        ]
    return A, b
end

# ╔═╡ 5c2a2f80-6884-4d72-914a-973a909c5a22
res = get_intrinsic_parameters(H_r)

# ╔═╡ 15bf4e50-3205-4663-bd85-d103e8611d87
b = res[2]

# ╔═╡ f235bd77-0cd0-4afb-a1ec-8572f3c01bba
B = [b[1] b[2] b[4];b[2] b[3] b[5];b[4] b[5] b[6]]

# ╔═╡ d1f50fcd-2be0-4d11-a433-137cddae14dd
A = res[1]

# ╔═╡ 72944130-7ed8-4873-9ca5-ba983f14d8df
# 13×2 Matrix{Matrix{Int32}}:
#  [244 94; 274 92; … ; 475 264; 510 266]    [0 0; 1 0; … ; 7 5; 8 5]
#  [256 357; 255 334; … ; 523 181; 539 131]  [0 0; 1 0; … ; 7 5; 8 5]
#  [277 72; 313 81; … ; 497 374; 544 390]    [0 0; 1 0; … ; 7 5; 8 5]
#  [188 130; 223 127; … ; 475 338; 521 338]  [0 0; 1 0; … ; 7 5; 8 5]
#  [435 50; 450 78; … ; 279 378; 288 431]    [0 0; 1 0; … ; 7 5; 8 5]
#  [589 138; 586 175; … ; 393 357; 389 387]  [0 0; 1 0; … ; 7 5; 8 5]
#  [368 137; 358 169; … ; 159 306; 151 334]  [0 0; 1 0; … ; 7 5; 8 5]
#  [470 92; 465 126; … ; 197 328; 184 370]   [0 0; 1 0; … ; 7 5; 8 5]
#  [219 85; 263 93; … ; 441 313; 469 313]    [0 0; 1 0; … ; 7 5; 8 5]
#  [413 65; 419 103; … ; 293 389; 301 429]   [0 0; 1 0; … ; 7 5; 8 5]
#  [423 71; 426 103; … ; 201 360; 198 408]   [0 0; 1 0; … ; 7 5; 8 5]
#  [402 71; 414 113; … ; 300 350; 311 374]   [0 0; 1 0; … ; 7 5; 8 5]
#  [415 57; 422 97; … ; 271 387; 279 422]    [0 0; 1 0; … ; 7 5; 8 5]

# ╔═╡ 91e78452-4a87-4d20-aa98-c4ae9310524d
begin
	img_num = 2
	λ = mean([1 /norm(inv(A) * H_r[img_num][:,1] ), 1 / norm(inv(A) * H_r[img_num][:,2] )])
	r1 = λ * inv(A) * H_r[img_num][:,1]
	r2 = λ * inv(A) * H_r[img_num][:,2]
	r3 = r1 .* r2
	t = λ * inv(A) * H_r[img_num][:,3]
	rt = [r1 r2 r3 t]

	img_cords, wor_cords =  chessboard_correspondences[img_num,:]
	@info img_cords, wor_cords
	projected = A * rt * [0, 0, 0, 1] # world point being projected to world, this matches data above
	projected = projected ./ projected[3]
end

# ╔═╡ d2823045-8b4e-447d-ae57-f26f42193327


# ╔═╡ Cell order:
# ╠═907ea08e-2d52-11ee-0218-67180aaaba6e
# ╠═0273fe28-3cf4-4d92-96ed-b4b81b789e8c
# ╠═c204a489-151d-4c37-8255-22803946a1c5
# ╠═e564e588-4336-4069-98bb-fb5cfa78a8f5
# ╠═933ec515-4bbe-4208-a496-221edc71b232
# ╠═bed0b3f8-83e6-4d7c-b54f-b9be0b5f71b9
# ╟─6b057b4c-7d75-48bd-9f9d-96d827618755
# ╠═0e990a5f-c9a4-4326-8aec-5f5147507a6b
# ╟─edbd55cf-8f54-4184-a7f2-5c805c98bdee
# ╠═bdff8656-8b51-4a33-912f-3b6128d757ed
# ╟─e5ffc1b5-f5ef-45b1-8387-8610825151e5
# ╠═1f668b5b-22aa-43d8-9293-fcf9c2bed5fe
# ╠═fe2c2081-67ab-4a6d-8d88-c81c42f3833a
# ╠═5c2a2f80-6884-4d72-914a-973a909c5a22
# ╠═15bf4e50-3205-4663-bd85-d103e8611d87
# ╠═f235bd77-0cd0-4afb-a1ec-8572f3c01bba
# ╠═d1f50fcd-2be0-4d11-a433-137cddae14dd
# ╠═72944130-7ed8-4873-9ca5-ba983f14d8df
# ╠═91e78452-4a87-4d20-aa98-c4ae9310524d
# ╠═d2823045-8b4e-447d-ae57-f26f42193327
