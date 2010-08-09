################################################################################
# Package Loading and Unloading
################################################################################

.onLoad <- function(libname, pkgname) {
   .C("magLoad", PACKAGE="magma")
}

.onUnload <- function(libname) {
   .C("magUnload", PACKAGE="magma")
}



