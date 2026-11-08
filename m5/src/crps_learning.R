# CRPS learning by BOA (profoc 1.3.4).  In "validation" mode every candidate in
# candidates.csv is run over the fitting origins; in "test" mode the selected
# candidate continues from its saved state over the test origins.
# Arguments: work directory, mode, extra R library ("" for none), engine file.
args <- commandArgs(trailingOnly=TRUE)
work <- normalizePath(args[1]); mode <- args[2]
if (nzchar(args[3])) .libPaths(c(normalizePath(args[3]), .libPaths()))
library(profoc)
source(normalizePath(args[4]))
stopifnot(as.character(packageVersion("profoc"))=="1.3.4")
meta <- jsonlite::read_json(file.path(work,"metadata.json"),simplifyVector=TRUE)
n <- as.integer(meta$n_series); p <- 99L; k <- length(meta$models)
n_fit <- as.integer(meta$n_fit); n_all <- length(meta$origins)
horizon <- as.integer(meta$horizon)
nt <- if(mode=="validation") n_fit else n_all
origins <- meta$origins[1:nt]
x <- lapply(1:nt,function(t) array(readBin(file.path(work,sprintf("x_%03d.bin",t-1)),
 "double",n=n*101*k,size=8),c(n,101,k)))
y <- matrix(readBin(file.path(work,if(mode=="validation") "y_validation.bin" else "y_all.bin"),
 "double",n=n*nt,size=8),n,nt)
spec <- read.csv(file.path(work,"candidates.csv"))
if(mode=="test") {
 selected <- jsonlite::read_json(file.path(work,"selection.json"),simplifyVector=TRUE)
 spec <- spec[spec$candidate_id==selected$candidate_id,,drop=FALSE]
}
for(j in seq_len(nrow(spec))) {
 s <- spec[j,]; id <- s$candidate_id
 if(mode=="validation") {
  result <- run_panel(x,y,origins,s$lambda,s$forget,horizon=horizon,last=n_fit)
  saveRDS(result$state,file.path(work,paste0(id,"_state.rds")))
 } else {
  state <- readRDS(file.path(work,paste0(id,"_state.rds")))
  result <- run_panel(x,y,origins,s$lambda,s$forget,horizon=horizon,state=state,
                      first=n_fit+1L,last=n_all)
 }
 writeBin(as.double(result$weights),file.path(work,paste0(id,"_",mode,"_weights.bin")))
 cat(sprintf("[profoc] %s %s (%d/%d) completed\n",mode,id,j,nrow(spec))); flush.console()
}
