rm -rf ../temp/compressed
mkdir -p ../temp/compressed
PATH=../jbigkit/pbmtools
$PATH/pbmtojbg $1 $2
echo "hello"
ls ../temp/compressed
zpaq a ../temp/compressed/$3 ../temp/compressed/$2 ../temp/res_$2
# $PATH/pbmtojbg m_im1.pbm ./compressed/m_im1.jbg
# zpaq a ./compressed/im1.archive ./compressed/m_im1.jbg res_im1.png