import re
text = "asdf          ds.!? $%^%&^^**,         "
text = re.sub(r"[^a-zA-Z0-9.?,!]"," ",text)
text = re.sub(' +',' ',text)
print(text)
