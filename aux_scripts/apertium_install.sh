#!/bin/bash

if [ $# -eq 0 ]; then
    echo -e "Needs one parameter: the path to the directory where apertium packages should be installed ";
    exit 1;
fi;
if ! [ -d "$1" ]; then
    echo -e "Passed parameter is not a valid path. ";
    exit 1;
fi;

locale-gen en_US.UTF-8 && update-locale;
curl -sS https://apertium.projectjj.com/apt/install-nightly.sh | bash;
apt install -y lttoolbox apertium-all-dev apertium-lex-tools apertium-separable hfst libhfst-dev cg3 cg3-dev autoconf libfoma0;

if [ $# -gt 1 ]; then

# Assuming we got a list of specific parsers to update.

PARSER_LIST=("${@:2}");

else

# Updating all parsers.

PARSER_LIST=("apertium-kaz" "apertium-tat" "apertium-rus" "apertium-kaz-rus" "apertium-tat-rus" "apertium-sah" "apertium-bak" "apertium-tat-bak" "apertium-tyv" "apertium-eng" "apertium-hye" "apertium-krc");

fi;

for PARSER_NAME in "${PARSER_LIST[@]}"; do

  if ! [ -d "$1/$PARSER_NAME" ]; then
    git clone https://github.com/apertium/$PARSER_NAME $1/$PARSER_NAME;
  else
    pushd $1/$PARSER_NAME && git pull && popd;
  fi;

  pushd $1/$PARSER_NAME && ./autogen.sh && ./configure && make && make install && popd;

done;

