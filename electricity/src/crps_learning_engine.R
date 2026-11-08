# BOA learner of profoc 1.3.4, updated with externally computed panel regrets.
# Quantile losses are averaged across series.
init_boa <- function(p, k, lambda, forget) {
  profoc::online(matrix(0), array(0, c(1, p, k)), (1:p)/(p+1),
    method="boa", loss_gradient=TRUE, lead_time=0, trace=FALSE,
    p_smooth_pr=list(knots=p, deg=3, ndiff=1.5, lambda=lambda),
    forget_regret=forget, gamma=1, regret=array(0,c(1,1,p,k)))
}

advance_boa <- function(model, regrets) {
  # update.online does not accept new external regrets, so they are passed to
  # profoc's internal C++ learner (hence the pinned version).
  p <- length(model$specification$data$tau)
  k <- dim(model$weights)[4]
  names <- model$specification$data$names
  model$weights <- profoc:::array_to_list(model$weights)
  model$past_performance <- profoc:::array_to_list(model$past_performance)
  model$experts_loss <- profoc:::array_to_list(model$experts_loss)
  engine <- new(profoc:::conline)
  engine$trace <- FALSE
  experts <- list(array(0,c(1,p,k))); dim(experts) <- c(1,1)
  engine$init_update(model, matrix(0), experts)
  dim(regrets) <- c(length(regrets),1)
  engine$regret_array <- regrets
  engine$learn()
  profoc:::post_process_model(engine, names)
}

last_weights <- function(model) {
  model$weights[dim(model$weights)[1],1,,]
}

run_panel <- function(x, y, origins, lambda, forget, horizon=28L, state=NULL,
                      first=1L, last=length(origins)) {
  n <- dim(x[[1]])[1]; p <- dim(x[[1]])[2]-2L; k <- dim(x[[1]])[3]
  tau <- (1:p)/(p+1)
  if (is.null(state)) {
    state <- list(model=init_boa(p,k,lambda,forget),
                  regrets=list(array(0,c(1,p,k))), issued=list(), next_r=1L)
  }
  weights <- array(NA_real_,c(last-first+1L,p,k))
  events <- list(); crossing <- 0L
  for (m in seq.int(first,last)) {
    while (state$next_r < m && origins[state$next_r]+horizon-1L < origins[m]) {
      r <- state$next_r
      stopifnot(all(is.finite(y[,r])), origins[r]+horizon-1L < origins[m])
      issued <- state$issued[[as.character(r)]]
      gradient <- (issued >= y[,r]) - matrix(tau,n,p,byrow=TRUE)
      regret <- matrix(0,p,k)
      for (j in 1:k) regret[,j] <- colMeans(gradient*(issued-matrix(x[[r]][,2:(p+1),j],n,p)))
      state$regrets[[length(state$regrets)+1L]] <- array(regret,c(1,p,k))
      state$model <- advance_boa(state$model, state$regrets)
      events[[length(events)+1L]] <- data.frame(current_origin=origins[m],
                                               realised_origin=origins[r])
      state$issued[[as.character(r)]] <- NULL
      state$next_r <- r+1L
    }
    w <- last_weights(state$model)
    stopifnot(all(is.finite(w)), min(w)>=0, max(abs(rowSums(w)-1))<1e-10)
    raw <- matrix(0,n,p)
    for (j in 1:k) raw <- raw + sweep(matrix(x[[m]][,2:(p+1),j],n,p),2,w[,j],"*")
    crossing <- crossing + sum(apply(raw,1,function(z) any(diff(z)<0)))
    issued <- t(apply(raw,1,sort))
    state$issued[[as.character(m)]] <- issued
    weights[m-first+1L,,] <- w
  }
  list(weights=weights, state=state, crossing=crossing,
       events=if(length(events)) do.call(rbind,events) else data.frame())
}
