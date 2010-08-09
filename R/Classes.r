################################################################################
# magma S4 Class
# Description:
#   Class for MAGMA-optimized matrix algebra operations
# Inherits:
#   matrix S3 class
# User-Specified Slots:
#   gpu - logical value to use GPU optomized algorithms
################################################################################

setClass("magma",
   representation(gpu = "logical"),
   prototype(gpu = TRUE),
   contains="matrix"
)

#################### Constructor ####################

magma <- function(data = NA, nrow = 1, ncol = 1, byrow = FALSE, dimnames = NULL,
                  gpu = TRUE)
{
   if(is(data, "magma")) {
      data@gpu <- gpu
      data
   } else if(is(data, "matrix")) {
      new("magma", data, gpu=gpu)
   } else {
      largs = list(data = data, byrow = byrow, dimnames = dimnames)
      if(!missing(nrow)) largs$nrow <- nrow
      if(!missing(ncol)) largs$ncol <- ncol
      new("magma", do.call("matrix", largs), gpu=gpu)
   }
}

#################### Coercion Functions ####################

setAs("magma", "matrix", function(from) from@.Data)
setAs("matrix", "magma", function(from) magma(from))
setAs("numeric", "magma", function(from) magma(from))

#################### General Methods ####################

setMethod("gpu", "magma", function(object) object@gpu)

setReplaceMethod("gpu", "magma",
  function(object, value) {
    object@gpu <- value
    object
	}
)

setMethod("show", "magma",
   function(object) {
      cat(nrow(object), "x", ncol(object), "magma matrix",
          ifelse(object@gpu, "with", "without"), "GPU optimization\n\n")
      print(as(object, "matrix"))
   }
)

setMethod("[", signature(x = "magma", i = "ANY", j = "ANY", drop = "ANY"),
   function(x, i, j, ..., drop) {
      y <- callGeneric(as(x, "matrix"), i=i, j=j, drop=drop)
      if(is.matrix(y)) magma(y, gpu=gpu(x)) else y
   }
)

setMethod("[", signature(x = "magma", i = "ANY", j = "missing", drop = "ANY"),
   function(x, i, j, ..., drop) {
      y <- callGeneric(as(x, "matrix"), i=i, , drop=drop)
      if(is.matrix(y)) magma(y, gpu=gpu(x)) else y
   }
)

setMethod("[", signature(x = "magma", i = "missing", j = "ANY", drop = "ANY"),
   function (x, i, j, ..., drop) {
      y <- callGeneric(as(x, "matrix"), , j=j, drop=drop)
      if(is.matrix(y)) magma(y, gpu=gpu(x)) else y
   }
)

setMethod("[", signature(x = "magma", i = "missing", j = "missing", drop = "ANY"),
   function(x, i, j, ..., drop) x
)


################################################################################
# magmaLU S4 Class
# Description:
#   Class for magma-generated LU decomposition
################################################################################

setClass("magmaLU",
   representation(pivot = "numeric"),
   prototype(pivot = integer(0)),
   contains = "magma"
)

#################### General Methods ####################

setMethod("show", "magmaLU",
   function(object) {
      cat("Object of class 'magmaLU'\n\n")
      print(as(object, "magma"))
   }
)


################################################################################
# magmaQR S4 Class
# Description:
#   Class for magma-generated QR decomposition
# Inherits:
#   qr S3 class
################################################################################

setOldClass("qr")
setClass("magmaQR",
   representation(work = "numeric"),
   prototype(structure(list(qr=magma(NA, 0, 0), rank=numeric(0),
                            qraux=numeric(0), pivot=numeric(0)),
                       useLAPACK=TRUE),
             work = numeric(0)),
   contains = "qr"
)

#################### General Methods ####################

setMethod("show", "magmaQR",
   function(object) {
      cat("Object of class 'magmaQR'\n\n")
      print(object[c("qr", "rank", "qraux", "pivot")])
   }
)

