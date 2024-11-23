PATH=../jbigkit/pbmtools
# zpaq x ./compressed/im1.archive
# $PATH/pbmtools/jbgtopbm ./compressed/m_im1.jbg m_im1.pbm
zpaq x ./compressed/$1
$PATH/jbgtopbm ./compressed/$2 $3