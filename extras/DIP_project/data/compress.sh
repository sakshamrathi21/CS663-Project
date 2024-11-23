rm -rf compressed
mkdir compressed
../../../jbigkit/pbmtools/pbmtojbg m_im2.pbm ./compressed/m_im2.jbg
zpaq a ./compressed/im2.archive ./compressed/m_im2.jbg res_im2.png