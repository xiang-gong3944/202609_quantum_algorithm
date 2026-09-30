#!/usr/bin/env perl

# カレントディレクトリ変更
$do_cd = 1;

# uplatexの呼び出し
$pdf_mode = 3;
$latex = 'uplatex -synctex=1 -file-line-error -halt-on-error %O %S';
$dvipdf = 'dvipdfmx %O -o %D %S';
$max_repeat = 5;

# bibtex系
$bibtex_use=2;
$bibtex = 'upbibtex %O %S';
$biber = 'biber --bblencoding=utf8 -u -U --output_safechars %O %S';

# index
$makeindex = 'upmendex %O -o %D %S -s jpbase';

# 出力フォルダ指定
$out_dir = "./out";

# 中間ファイル登録
$clean_ext="$clean_ext run.xml";
