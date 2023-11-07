style="<style> body {font-family: arial;}<\/style>"
anchor='<link rel="icon" href="u1F367-shavedice.svg">'
anchor2='<img src="u1F367-shavedice.svg" alt="">'
find -name '*.html' | xargs sed -i "s/$anchor$/$anchor$style/g"
find -name '*.html' | xargs sed -i "s/$anchor2 ki$/$anchor2/g"
