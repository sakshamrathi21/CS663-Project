rm -rf compressed
mkdir compressed
../../../jbigkit/pbmtools/pbmtojbg m_im1.pbm ./compressed/m_im1.jbg
zpaq a ./compressed/im1.archive ./compressed/m_im1.jbg res_im1.png