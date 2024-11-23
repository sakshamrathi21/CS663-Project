rm -rf compressed
mkdir compressed
../../../jbigkit/pbmtools/pbmtojbg m_im3.pbm ./compressed/m_im3.jbg
zpaq a ./compressed/im3.archive ./compressed/m_im3.jbg res_im3.png