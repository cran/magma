################################################################################
# Arithmetic Operatorations
################################################################################


#################### Generic Arithmetic Operators ####################

setMethod("Arith", signature(e1 = "magma", e2 = "magma"),
   function(e1, e2)
      magma(callGeneric(as(e1, "matrix"), as(e2, "matrix")), gpu=gpu(e1))
)

setMethod("Arith", signature(e1 = "magma", e2 = "matrix"),
   function(e1, e2) magma(callGeneric(as(e1, "matrix"), e2), gpu=gpu(e1))
)

setMethod("Arith", signature(e1 = "magma", e2 = "numeric"),
   function(e1, e2) magma(callGeneric(as(e1, "matrix"), e2), gpu=gpu(e1))
)

setMethod("Arith", signature(e1 = "matrix", e2 = "magma"),
   function(e1, e2) magma(callGeneric(e1, as(e2, "matrix")), gpu=gpu(e2))
)

setMethod("Arith", signature(e1 = "numeric", e2 = "magma"),
   function(e1, e2) magma(callGeneric(e1, as(e2, "matrix")), gpu=gpu(e2))
)


#################### Multiplication Operators ####################

setMethod("%*%", signature(x = "magma", y = "magma"),
   function(x, y) .Call("magMultmm", x, FALSE, y, FALSE)
)

setMethod("%*%", signature(x = "magma", y = "matrix"),
   function(x, y) .Call("magMultmm", x, FALSE, y, FALSE)
)

setMethod("%*%", signature(x = "magma", y = "numeric"),
   function(x, y) {
      if(ncol(x) == 1) .Call("magMultmm", x, FALSE, as.matrix(y), TRUE)
      else .Call("magMultmv", x, FALSE, y, TRUE)
   }
)

setMethod("%*%", signature(x = "matrix", y = "magma"),
   function(x, y) .Call("magMultmm", x, FALSE, y, FALSE)
)

setMethod("%*%", signature(x = "numeric", y = "magma"),
   function(x, y) {
      if(nrow(y) == 1) .Call("magMultmm", as.matrix(x), FALSE, y, FALSE)
      else .Call("magMultmv", y, FALSE, x, FALSE)
   }
)


#################### Crossproduct Functions: t(A) %*% B ####################

setMethod("crossprod", signature(x = "magma", y = "magma"),
   function(x, y) .Call("magMultmm", x, TRUE, y, FALSE)
)

setMethod("crossprod", signature(x = "magma", y = "matrix"),
   function(x, y) .Call("magMultmm", x, TRUE, y, FALSE)
)

setMethod("crossprod", signature(x = "magma", y = "missing"),
   function(x, y) .Call("magMultmm", x, TRUE, x, FALSE)
)

setMethod("crossprod", signature(x = "magma", y = "numeric"),
   function(x, y) .Call("magMultmv", x, TRUE, y, TRUE)
)

setMethod("crossprod", signature(x = "matrix", y = "magma"),
   function(x, y) .Call("magMultmm", x, TRUE, y, FALSE)
)

setMethod("crossprod", signature(x = "numeric", y = "magma"),
   function(x, y) .Call("magMultmv", y, FALSE, x, FALSE)
)


#################### tCrossproduct Functions: A %*% t(B) ####################

setMethod("tcrossprod", signature(x = "magma", y = "magma"),
   function(x, y) .Call("magMultmm", x, FALSE, y, TRUE)
)

setMethod("tcrossprod", signature(x = "magma", y = "matrix"),
   function(x, y) .Call("magMultmm", x, FALSE, y, TRUE)
)

setMethod("tcrossprod", signature(x = "magma", y = "missing"),
   function(x, y) .Call("magMultmm", x, FALSE, x, TRUE)
)

setMethod("tcrossprod", signature(x = "magma", y = "numeric"),
   function(x, y) .Call("magMultmm", x, FALSE, as.matrix(y), TRUE)
)

setMethod("tcrossprod", signature(x = "matrix", y = "magma"),
   function(x, y) .Call("magMultmm", x, FALSE, y, TRUE)
)

setMethod("tcrossprod", signature(x = "numeric", y = "magma"),
   function(x, y) {
      if(ncol(y) == 1) .Call("magMultmm", as.matrix(x), FALSE, y, TRUE)
      else .Call("magMultmv", y, TRUE, x, FALSE)
   }
)

