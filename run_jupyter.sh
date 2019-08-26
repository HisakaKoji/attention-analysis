#!/bin/sh

mkdir -p ./model/bert-wiki-ja

if [ ! -e ./model/bert-wiki-ja/graph.pbtxt ]; then
  wget "https://drive.google.com/uc?export=download&id=11V3dT_xJUXsZRuDK1kXiXJRBSEHGl3In" -O ./model/bert-wiki-ja/graph.pbtxt
fi

if [ ! -e ./model/bert-wiki-ja/model.ckpt-1400000.data-00000-of-00001 ]; then
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1F4b_u-5zzqabA6OfLxDkLh0lzqVIEZuN" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1F4b_u-5zzqabA6OfLxDkLh0lzqVIEZuN" -o ./model/bert-wiki-ja/model.ckpt-1400000.data-00000-of-00001
fi

if [ ! -e ./model/bert-wiki-ja/model.ckpt-1400000.index ]; then
  wget "https://drive.google.com/uc?export=download&id=1LB00MDQJjb-xLmgBMhdQE3wKDOLjgum-" -O ./model/bert-wiki-ja/model.ckpt-1400000.index
fi

if [ ! -e ./model/bert-wiki-ja/model.ckpt-1400000.meta ]; then
  wget "https://drive.google.com/uc?export=download&id=1V9TIUn5wc-mB_wabYiz9ikvLsscONOKB" -O ./model/bert-wiki-ja/model.ckpt-1400000.meta
fi

if [ ! -e ./model/bert-wiki-ja/wiki-ja.model ]; then
  wget "https://drive.google.com/uc?export=download&id=1jjZmgSo8C9xMIos8cUMhqJfNbyyqR0MY" -O ./model/bert-wiki-ja/wiki-ja.model
fi

if [ ! -e ./model/bert-wiki-ja/wiki-ja.vocab ]; then
  wget "https://drive.google.com/uc?export=download&id=1uzPpW38LcS4YS431GgdG0Hsj4gNgE5X1" -O ./model/bert-wiki-ja/wiki-ja.vocab
fi

docker build -t attention-analysis ./docker
docker run --rm -it --name=attention-analysis -p 8888:8888 -v `pwd`:/work/attention-analysis attention-analysis

