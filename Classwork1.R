###########################################
# Step-by-step K-means Clustering on Iris #
# Author: Rento Saijo                     #
###########################################

# This script teaches K-means by showing each iteration on a scatter plot.
# The title of the plot shows SSE (sum of squared errors), which we want to shrink over time.

# ---------- User controls ----------
K <- 3
# K is the number of clusters we want the algorithm to find.

max_iter <- 25
# This is a safety limit so the loop cannot run forever.

tol <- 1e-6
# This is the stopping threshold.
# If the centroids barely move (less than tol), we will stop.

seed <- 123
# This makes the random choices repeatable, so you get the same result every time.

pause_seconds <- 1.5
# This pauses between iterations so you can watch the algorithm update.

vis_features <- c('Petal.Length', 'Petal.Width')
# These are the two features (columns) we will show on the 2D plot.

cluster_features <- c('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width')
# These are the features we actually use for clustering (all 4 numeric iris columns).

standardize <- TRUE
# If TRUE, we scale each feature to have mean 0 and standard deviation 1.
# This prevents features with larger numeric ranges from dominating the distance.

# ---------- Setup ----------
set.seed(seed)
# This sets the random seed.

X_raw <- as.matrix(iris[, cluster_features])
# We convert the selected columns into a numeric matrix.

n <- nrow(X_raw)
# n is the number of data points (rows).

if (standardize) {
  X <- scale(X_raw)  # clustering space
  # 'scale' standardizes each column: (value - mean) / standard_deviation.
  # We do clustering in this scaled space.
} else {
  X <- X_raw
  # If we do not standardize, we cluster in the original units.
}

# Plot in raw units for readability
X_vis <- as.matrix(iris[, vis_features])
# We plot using the original units so the axis labels match the iris dataset.

fmt <- function(x, digits = 6) format(round(x, digits), nsmall = digits)
# Small helper function to print numbers nicely.

# ---------- Linear Algebra Helpers ----------

# This function computes the squared distance from every point to every centroid.
# Output: an n x K table (matrix) of squared distances.
# D2[i, k] = ||x_i - c_k||^2
#
# ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x · c)
# This lets us compute all distances using matrix multiplication.

sq_dist_matrix <- function(X, C) {
  X2 <- rowSums(X^2)
  # X2[i] is ||x_i||^2, the squared length of point i.

  C2 <- rowSums(C^2)
  # C2[k] is ||c_k||^2, the squared length of centroid k.

  D2 <- outer(X2, C2, '+') - 2 * (X %*% t(C))
  # outer(X2, C2, '+') creates an n x K matrix where entry (i,k) is X2[i] + C2[k].
  # X %*% t(C) is matrix multiplication that gives all dot products x_i · c_k.
  # Putting it together gives all squared distances at once.

  D2[D2 < 0] <- 0
  # Tiny negative values can happen because of floating-point rounding.
  # Squared distances should never be negative, so we clamp them to 0.

  D2
}

# This function creates a "membership matrix" M (n x K).
# M[i,k] = 1 means point i belongs to cluster k, and 0 otherwise.
# This is sometimes called a one-hot encoding.

membership_matrix <- function(assign, K) {
  M <- matrix(0, nrow = length(assign), ncol = K)
  # Start with all zeros.

  M[cbind(seq_along(assign), assign)] <- 1
  # Put a 1 in the correct cluster column for each row.

  M
}

# This function updates the centroid locations after assignments are made.
#
# - t(M) %*% X gives the sum of points in each cluster (a K x d matrix).
# - colSums(M) gives the number of points in each cluster (length K).
# - centroid = (cluster sum) / (cluster count)

update_centroids <- function(X, M) {
  counts <- colSums(M)
  # counts[k] is how many points are currently in cluster k.

  sums <- t(M) %*% X
  # sums[k, ] is the sum of all points assigned to cluster k.

  C_new <- sums
  # We will turn sums into means (averages) by dividing by counts.

  for (k in 1:nrow(C_new)) {
    if (counts[k] > 0) C_new[k, ] <- sums[k, ] / counts[k]
    else C_new[k, ] <- NA
    # If a cluster has zero points, its centroid is undefined.
    # We mark it as NA for now and fix it later.
  }

  list(C = C_new, counts = counts, sums = sums)
}

# This function converts centroids (which might be in scaled space) into raw units
# so we can plot them correctly on the same axes as X_vis.

centroids_to_vis <- function(C_std_or_raw, X_scaled) {
  if (!standardize) {
    # If we clustered in raw space, the centroid values are already in raw units.
    C_vis <- C_std_or_raw[, match(vis_features, cluster_features), drop = FALSE]
  } else {
    centers <- attr(X_scaled, 'scaled:center')
    scales  <- attr(X_scaled, 'scaled:scale')
    # These are the means and standard deviations used by scale().

    C_raw <- sweep(C_std_or_raw, 2, scales, '*')
    C_raw <- sweep(C_raw, 2, centers, '+')
    # This "unscales" the centroids back into the original units.

    C_vis <- C_raw[, match(vis_features, cluster_features), drop = FALSE]
  }
  C_vis
}

# ---------- Visualization ----------
# This function draws a scatter plot and shows the current centroid locations.
# The plot title includes SSE so you can see the objective going down.

plot_state <- function(iter, X_vis, assign, C_vis, sse) {
  palette <- c('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
               '#e6ab02', '#a6761d', '#666666')
  # A list of colors. We will use one color per cluster.

  cols <- palette[assign]
  # Each data point gets the color of its assigned cluster.

  plot(
    X_vis[, 1], X_vis[, 2],
    col = cols, pch = 19, cex = 1.0,
    xlab = vis_features[1], ylab = vis_features[2],
    main = paste0('K-means | Iter ', iter, ' | K=', K, ' | SSE=', fmt(sse))
  )
  # pch = 19 means solid circles.

  points(C_vis[, 1], C_vis[, 2], pch = 8, cex = 2.0, lwd = 2)
  # pch = 8 makes a star-like marker for centroids.

  text(C_vis[, 1], C_vis[, 2], labels = paste0('C', 1:K), pos = 3, cex = 0.9)
  # Label each centroid C1, C2, ..., CK.

  legend(
    'bottomright',
    legend = paste0('Cluster ', 1:K),
    col = palette[1:K],
    pch = 19,
    cex = 0.9,
    bty = 'n'
  )
  # This legend explains the cluster colors.
  # bty = 'n' removes the legend box for a cleaner look.
}

# ---------- Initialize Centroids ----------
# We start by picking K random data points to be the first centroids.
# This is a common simple initialization.

init_idx <- sample(1:n, K, replace = FALSE)
C <- X[init_idx, , drop = FALSE]

cat('=== Initialization ===\n')
cat('Selected initial centroid indices:', paste(init_idx, collapse = ', '), '\n')

# Initial assignment + SSE
D2 <- sq_dist_matrix(X, C)
# D2[i,k] contains the squared distance from point i to centroid k.

assign <- max.col(-D2)
# max.col chooses the index of the largest value in each row.
# We use -D2 so that the smallest distance becomes the largest negative distance.
# Result: assign[i] is the cluster number (1..K) for point i.

sse <- sum(D2[cbind(1:n, assign)])
# SSE = sum over i of the squared distance from point i to its chosen centroid.
# This is the number we want to decrease.

cat('Initial SSE =', fmt(sse), '\n')

# Initial plot
C_vis <- centroids_to_vis(C, X)
plot_state(iter = 0, X_vis = X_vis, assign = assign, C_vis = C_vis, sse = sse)
if (pause_seconds > 0) Sys.sleep(pause_seconds)
# This pause is just to give you time to see the first picture.

# ---------- Iterate ----------
# Each loop does one full K-means iteration:
# 1) Compute distances
# 2) Assign points
# 3) Update centroids

for (iter in 1:max_iter) {
  cat('\n========================================\n')
  cat('Iteration', iter, '\n')
  cat('========================================\n')

  # (1) Distances
  D2 <- sq_dist_matrix(X, C)
  # We compute every squared distance from every point to every centroid.

  # (2) Assignment
  assign_new <- max.col(-D2)
  # Each point goes to the nearest centroid (smallest squared distance).

  # (2b) SSE with current centroids and the new assignments
  sse <- sum(D2[cbind(1:n, assign_new)])
  cat('SSE =', fmt(sse), '\n')
  # Watching SSE helps you see if we are improving.

  # (3) Membership matrix
  M <- membership_matrix(assign_new, K)
  # This turns the assignments into a one-hot matrix.

  # (4) Centroid update
  upd <- update_centroids(X, M)
  C_new <- upd$C
  # Each centroid becomes the average of the points assigned to it.

  # Empty cluster fix: reinitialize centroid to a random point
  if (any(is.na(C_new))) {
    empty <- which(is.na(C_new[, 1]))
    cat('WARNING: Empty cluster(s):', paste(empty, collapse = ', '), '\n')
    cat('We will reinitialize those centroid(s) to a random data point.\n')
    for (k in empty) {
      C_new[k, ] <- X[sample(1:n, 1), ]
    }
  }

  # (5) Convergence check: centroid movement
  movement <- sqrt(rowSums((C_new - C)^2))
  # movement[k] is how far centroid k moved in this iteration.

  cat('Centroid movement:', paste0('C', 1:K, '=', fmt(movement), collapse = '  '), '\n')

  # Commit updates
  C <- C_new
  assign <- assign_new
  # We overwrite old centroids and assignments with the new ones.

  # Visualize current iteration (single plot, SSE in title)
  C_vis <- centroids_to_vis(C, X)
  plot_state(iter = iter, X_vis = X_vis, assign = assign, C_vis = C_vis, sse = sse)
  if (pause_seconds > 0) Sys.sleep(pause_seconds)

  # Stop early if centroids are no longer moving in a meaningful way.
  if (max(movement) < tol) {
    cat('\nConverged: max centroid movement <', tol, '\n')
    break
  }
}

# ---------- Compare to Built-in K-means ----------
# This is a quick sanity check using R's built-in K-means function.
# It should reach a similar SSE (not always identical because initialization can differ).

cat('\n=== Built-in kmeans() comparison (validation) ===\n')
km <- kmeans(X, centers = K, nstart = 10, iter.max = 100)
# nstart = 10 means it tries 10 random initializations and keeps the best one.

cat('kmeans() cluster sizes:', paste(km$size, collapse = ', '), '\n')
cat('kmeans() tot.withinss (SSE):', fmt(km$tot.withinss), '\n')

cat('\nDone.\n')
