################################################################################
# Matrix Factorizations
################################################################################


#################### Cholesky Factorization ####################

setMethod("chol", signature(x = "magma"),
   function(x) .Call("magChol", x)
)


#################### LU Factorization ####################

setMethod("lu", signature(x = "magma"),
   function(x, ...) .Call("magLU", x)
)


#################### QR Factorization ####################

setMethod("qr", signature(x = "magma"),
   function(x, ...) .Call("magQR", x)
)

