import re
import lxml.html
import lxml.html.clean


def cleanhtml(raw_html):
    if raw_html is None: 
        return ''
    else:        
        #cleanr = re.compile('<.*?>')
        #cleantext = re.sub(cleanr, '', raw_html)
        #cleantext = cleantext.replace('<!--*/','<')
        
        cleanr = re.compile("(?s)<style type=\"text/css\">.*?</style>")
        cleantext = re.sub(cleanr, '', raw_html)  
 
        
        cleanr = re.compile("(?s)<.*?>")
        cleantext = re.sub(cleanr, '', cleantext)        

        cleantext = cleantext.replace('*/-->','')

       
        doc = lxml.html.fromstring(cleantext)
        cleaner = lxml.html.clean.Cleaner(style=True)
        doc = cleaner.clean_html(doc)
        text = doc.text_content()

        return text