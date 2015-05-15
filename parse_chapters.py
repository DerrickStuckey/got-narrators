__author__ = 'dstuckey'

#open the first book

class Chapter:
    def __init__(self, name, content):
        self.name = name
        self.content = content

#book_filename = 'raw_text/#1A Game of Thrones.txt'
#book_filename = 'raw_text/#2A Clash of Kings.txt'
#book_filename = 'raw_text/#3A Storm of Swords.txt'
#book_filename = 'raw_text/#4A Feast for Crows.txt'
#book_filename = 'raw_text/#5A Dance With Dragons.txt'
book_filename = 'raw_text/#5A Dance With Dragons cleaned.txt'

def parse_chapters(book_filename):
    chapter_names = []
    chapters = []

    with open(book_filename) as book:
        current_chapter = ""
        for line in book.readlines():
            #remove non-ascii characters
            line = ''.join(i for i in line if ord(i)<128)

            #break at appendix
            if ("APPENDIX" in line or "EPILOGUE" in line):
                break

            if (line.isupper() and not (" " in line.strip())):
                # print line
                chapter_names.append(line.strip())
                chapters.append(current_chapter)
                current_chapter = ""
            else:
                #chapters[len(chapters)-1] = chapters[len(chapters)-1] + line
                current_chapter = current_chapter + line

        #now add the last chapter, and remove the first (pre-prologue text)
        chapters.append(current_chapter)
        chapters.pop(0)

    zipped = zip(chapter_names, chapters)

    return [Chapter(z[0],z[1]) for z in zipped]

# chaps = parse_chapters(book_filename)
# for chap in chaps:
#     print "Chapter: ", chap.name
#     print chap.content[0:50]