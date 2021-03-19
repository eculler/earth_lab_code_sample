#!/opt/local/bin/bash
# Configure paths
LOCALWORKDIR=~/Documents/research/landslide.hydromet.git/preprocess
LOCALDATADIR=/Volumes/WD-Data/data/by.project/landslide.hydromet.data
LOCALOUTDIR=~/GoogleDrive/research/landslide.hydromet.sync/data/
LOCALSRCDIR=${LOCALWORKDIR}/src
DOCKERWORKDIR=/hydromet
DOCKERDATADIR=${DOCKERWORKDIR}/data
DOCKERSRCDIR=${DOCKERWORKDIR}/src
DOCKEROUTDIR=${DOCKERWORKDIR}/out

PRECIPNAME="$1"
LOGLEVEL=debug
LON_NAME=lon
LAT_NAME=lat
TIME_NAME=time
X_NAME=$LON_NAME
Y_NAME=$LAT_NAME
TO360=0
PRECIPCRS='EPSG:4326'
ENGINE=netcdf4

# Build container
#cd ${LOCALWORKDIR} && docker build --tag=hydromet ${LOCALWORKDIR}

for YEAR in {2018..2018}; do
  case ${PRECIPNAME} in
    nldas)
      PRECIP="precipitation/NLDAS2/NLDAS_FORA0125_H.A${YEAR}\*.\*.002.grb.SUB.nc4"
      PRECIP_NAME=APCP
      ;;
    imergf)
      PRECIP="precipitation/IMERGF/${YEAR}/3B-HHR.MS.MRG.3IMERG.${YEAR}\*-S\*.nc4"
      PRECIP_NAME=precipitationCal
      ;;
    imerge)
      PRECIP="precipitation/IMERGE/${YEAR}/3B-HHR-E.MS.MRG.3IMERG.${YEAR}\*-S\*.nc4"
      PRECIP_NAME=precipitationCal
      ;;
    mrms)
      PRECIP="precipitation/MRMS/"
      PRECIP_NAME=precip
      ;;
    hrrr)
      PRECIP="precipitation/HRRR/hrrr_qpf_${YEAR}/${YEAR}1013/hrrrx_qpf_${YEAR}1013\*.grib2"
      PRECIP_NAME="tp"
      ENGINE="cfgrib"
      LON_NAME="longitude"
      LAT_NAME="latitude"
      X_NAME="x"
      Y_NAME="y"
      TO360=1
      ;;
  esac

  # Start container, run python script
  set -f && docker run --rm -it \
    --security-opt seccomp=unconfined \
    -v ${LOCALDATADIR}:${DOCKERDATADIR} `# Input data` \
    -v ${LOCALSRCDIR}:${DOCKERSRCDIR} `# Processing code` \
    -v ${LOCALOUTDIR}:${DOCKEROUTDIR} `# Output directory` \
    -t hydromet:latest \
    conda run -n hydrometenv \
    python ${DOCKERSRCDIR}/landslide_precip.py \
    ${DOCKERDATADIR}/${PRECIP} \
    ${DOCKERDATADIR}/landslide/landslides.verified.csv \
    ${DOCKEROUTDIR}/${PRECIPNAME}.${YEAR}.csv \
    ${LON_NAME} \
    ${LAT_NAME} \
    ${PRECIP_NAME} \
    ${TIME_NAME} \
    ${X_NAME} \
    ${Y_NAME} \
    ${TO360} \
    ${ENGINE} \
    "${PRECIPCRS}" \
    ${LOGLEVEL}
done
