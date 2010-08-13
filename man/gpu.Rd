\name{gpu}
\alias{gpu}
\alias{gpu<-}
\alias{gpu,magma-method}
\alias{gpu<-,magma-method}

\title{GPU Optimization for magma Matrices}

\description{
Retrieve or set the flag for GPU (graphics processing unit) optimization of an \linkS4class{magma} matrix.
}

\usage{
\S4method{gpu}{magma}(object)
\S4method{gpu}{magma}(object) <- value
}

\arguments{
  \item{object}{a \linkS4class{magma} class object.}
  \item{value}{logical.  If \code{TRUE}, gpu-optimized parallel algorithms are
  utilized for matrix operations, when available.  Otherwise, more
  general-purpose parallel algorithms are used.}
}

\author{
Brian J. Smith <brian-j-smith@uiowa.edu>
}

\seealso{
\code{\linkS4class{magma}}
}

\examples{
x <- magma(1:6, nrow=2, ncol=3)
gpu(x)

gpu(x) <- !gpu(x)
gpu(x)
}

\keyword{classes}

