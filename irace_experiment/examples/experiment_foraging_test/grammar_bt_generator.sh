#!/bin/bash

# Syntax printing
function print_syntax() {
    echo
    echo "To generate the txt with the descrition of the possible grammar for the behavior tree"
    echo "specify MAX_NBR_BRANCHES, MAX_NBR_CONDITIONS (the number of conditions within the selector subtrees),"
    echo "and the txt file name in which the description will be saved:"
    echo "$0 <MAX_NBR_BRANCHES> <MAX_NBR_CONDITIONS> <TXT_FILE>"
    echo
    exit 1
}

# Write description of a branch
function write_branch() {
  INDEX=$1
  NBR_CONDITIONS=$(echo "$2-1" | bc)
  echo "N$INDEX     \"--n$INDEX \"  c   (0) | as.numeric(NumChildsRoot)>$INDEX " >> ${TXT_FILE}
  echo "NumConds$INDEX   \"--nc$INDEX \"  i (1,$2) | as.numeric(N$INDEX)==0" >> ${TXT_FILE}

  for CUR_COND in $(seq 0 $NBR_CONDITIONS)
  do
    echo "BRANCH $INDEX COND $CUR_COND OUT OF $NBR_CONDITIONS"
    write_condition $INDEX $CUR_COND
  done
  write_action $INDEX

}

function write_action() {
  INDEX=$1
  echo "ACTION $1"
  echo "Action$INDEX     \"--a$INDEX \"  c   (0,1,2,3,4,5) | as.numeric(N$INDEX)==0 " >> ${TXT_FILE}
  echo "RWM$INDEX   \"--rwm$INDEX \"  i (1,100) | as.numeric(Action$INDEX)==0" >> ${TXT_FILE}
  echo "ATT$INDEX   \"--att$INDEX \"  r (1,5) | as.numeric(Action$INDEX)==4" >> ${TXT_FILE}
  echo "REP$INDEX   \"--rep$INDEX \"  r (1,5) | as.numeric(Action$INDEX)==5" >> ${TXT_FILE}
}

function write_condition() {
  BRANCH=$1
  COND=$2
  echo "C${BRANCH}x$COND  \"--c${BRANCH}x$COND \" c   (0,1,2,3,4,5) | as.numeric(NumConds$BRANCH)>$COND " >> ${TXT_FILE}
  echo "P${BRANCH}x$COND  \"--p${BRANCH}x$COND \" r   (0,1) | as.numeric(C${BRANCH}x$COND) %in% c(0,1,2,5) " >> ${TXT_FILE}
  echo "B${BRANCH}x$COND  \"--p${BRANCH}x$COND \" i   (1,10) | as.numeric(C${BRANCH}x$COND)==3 " >> ${TXT_FILE}
  echo "W${BRANCH}x$COND  \"--w${BRANCH}x$COND \" r   (0,20) | as.numeric(C${BRANCH}x$COND)==3 " >> ${TXT_FILE}
  echo "BI${BRANCH}x$COND  \"--p${BRANCH}x$COND \" i   (1,10) | as.numeric(C${BRANCH}x$COND)==4 " >> ${TXT_FILE}
  echo "WI${BRANCH}x$COND  \"--w${BRANCH}x$COND \" r   (0,20) | as.numeric(C${BRANCH}x$COND)==4 " >> ${TXT_FILE}
}

if [ $# -lt 3 ]; then
    print_syntax
fi

MAX_NBR_BRANCHES=$(echo "$1-1" | bc)
TXT_FILE=$3

# Clear content of file
truncate -s 0 $TXT_FILE

# Write grammar in file
echo "RootNode   \"--rootnode \"   c (0)" >> $TXT_FILE
echo "NumChildsRoot   \"--nchildsroot \"   i (1,$1)" >> $TXT_FILE
for BRANCH in $(seq 0 $MAX_NBR_BRANCHES)
do
  write_branch $BRANCH $2
done
