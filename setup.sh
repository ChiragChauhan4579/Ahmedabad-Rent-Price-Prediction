mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "[global]
showWarningOnDirectExecution = false
[theme]
primaryColor = '#f21111'
backgroundColor='#0e1117'
secondaryBackgroundColor='#31333F'
textColor='#fafafa'
font='sans serif'
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml