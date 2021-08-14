PACMENSL_VERSION=0.1.0
ARCH=$(uname -m)

docker build -t pacmensl:v${PACMENSL_VERSION}_${ARCH} --build-arg pacmensl_version=$PACMENSL_VERSION .