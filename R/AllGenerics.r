################################################################################
# Generic Class Method Declarations
################################################################################

setGeneric("backsolve",
           function(r, x, k = ncol(r), upper.tri = TRUE, transpose = FALSE)
           standardGeneric("backsolve"))
setGeneric("gpu", function(object) standardGeneric("gpu"))
setGeneric("gpu<-", function(object, value) standardGeneric("gpu<-"))
setGeneric("lu", function(x, ...) standardGeneric("lu"))

