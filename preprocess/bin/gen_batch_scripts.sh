#!/bin/bash
PRECIPNAME="$1"
LOGLEVEL=debug
TEMPLATE="$2"
LON_NAME=lon
LAT_NAME=lat
TIME_NAME=time
X_NAME=$LON_NAME
Y_NAME=$LAT_NAME
TO360='false'
PRECIPCRS='EPSG:4326'
ENGINE=netcdf4

# Configure paths
WORKDIR=/projects/elcu4811/landslide.hydromet/landslide.hydromet.git/preprocess
DATADIR=/scratch/summit/elcu4811/landslide.hydromet.data
OUTDIR=/projects/elcu4811/landslide.hydromet/landslide.hydromet.out/$PRECIPNAME
SRCDIR=${WORKDIR}/src
BATCHDIR=${SRCDIR}/batch
if [ $TEMPLATE == "test" ]
then
  TEMPLATE_PATH="batch_template_test.txt"
  MONTH=10
  DAY=01
  STARTYEAR=2018
  ENDYEAR=2018
else
  TEMPLATE_PATH="batch_template.txt"
  MONTH=""
  DAY=""
  STARTYEAR=2014
  ENDYEAR=2020
fi
mkdir -p $WORKDIR $OUTDIR $BATCHDIR

# Make batch script for each year
for ((YEAR=STARTYEAR; YEAR<=ENDYEAR; YEAR++)); do
  for ((MON=1; MON<=12; MON++)); do

MONTH=$(printf "%02d" $MON)
BATCHPATH=${BATCHDIR}/${PRECIPNAME}.${TEMPLATE}.${YEAR}${MONTH}.batch
case ${PRECIPNAME} in
  nldas)
    PRECIP="precipitation/NLDAS2/NLDAS_FORA0125_H.A${YEAR}${MONTH}\*.\*.002.grb.SUB.nc4"
    PRECIP_NAME=APCP
    ;;
  imergf)
    PRECIP="precipitation/IMERGF/${YEAR}/3B-HHR.MS.MRG.3IMERG.${YEAR}${MONTH}\*-S\*.nc4"
    PRECIP_NAME=precipitationCal
    ;;
  imerge)
    PRECIP="precipitation/IMERGE/${YEAR}/3B-HHR-E.MS.MRG.3IMERG.${YEAR}${MONTH}\*-S\*.nc4"
    PRECIP_NAME=precipitationCal
    ;;
  mrms)
    PRECIP="precipitation/MRMS.1hour/GaugeCorr_QPE_01H_00.00_${YEAR}${MONTH}${DAY}\*.grib2"
    PRECIP_NAME=paramId_0
    LON_NAME='longitude'
    LAT_NAME='latitude'
    X_NAME=$LON_NAME
    Y_NAME=$LAT_NAME
    TO360='true'
    ENGINE="cfgrib"
    ;;
  hrrr)
    PRECIP="precipitation/HRRR/${YEAR}/${YEAR}${MONTH}\*/hrrrx_qpf_${YEAR}${MONTH}\*.grib2"
    PRECIP_NAME="tp"
    ENGINE="cfgrib"
    LON_NAME="longitude"
    LAT_NAME="latitude"
    X_NAME="x"
    Y_NAME="y"
    TO360=true
    ;;
esac

# Initialize file
echo $BATCHPATH
cp ${SRCDIR}/${TEMPLATE_PATH} ${BATCHPATH}

# Write command to start container, run python script
cat >> ${BATCHPATH} << EOL
singularity run --bind /scratch/summit ${WORKDIR}/hydrological.processes.202011.sif bash -c \\
  'conda activate hydrometenv && python ${SRCDIR}/landslide_precip.py ${DATADIR}/${PRECIP} ${DATADIR}/landslide/landslides.verified.csv ${OUTDIR}/${PRECIPNAME}.${YEAR}${MONTH}.csv ${LON_NAME} ${LAT_NAME} ${PRECIP_NAME} ${TIME_NAME} ${X_NAME} ${Y_NAME} ${TO360} ${ENGINE} "${PRECIPCRS}" ${LOGLEVEL}'
EOL

done
done
