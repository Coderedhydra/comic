import os
import random
import copy
from backend.class_def import panel


template_specs = {
    "1" : {
        "span" : 1,
        "direction": "row"
    },
    "2" : {
        "span" : 2,
        "direction": "row"
    },
    "3" : {
        "span" : 3,
        "direction": "column"
    },
     "4" : {
        "span" : 2,
        "direction": "column"
    },
    # High-accuracy templates with fewer, larger panels
    "5" : {
        "span" : 4,
        "direction": "row"
    },
    "6" : {
        "span" : 2,
        "direction": "row"
    },
    "7" : {
        "span" : 3,
        "direction": "row"
    },
    "8" : {
        "span" : 4,
        "direction": "column"
    }
}

input = '433343333343343333443333443334333343344443433'



def hammingDist(str1, str2): 
    i = 0
    count = 0
  
    while(i < len(str1)): 
        if(str1[i] != str2[i]): 
            count += 1
        i += 1
    return count

def get_files_in_folder(folder_path):
    file_dicts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            rank = random.randint(1, 3) 

            file_dicts.append({"name": file , 'rank' :  rank})
    return file_dicts

templates = ['14124114','312341' , '4432111' , '21411241' , '3241141' , '13411141' , '12411131' ,'1321113', '131423' , 
'142344' , '234241','2411413','3141214','42111131']

# High-accuracy mode: when HIGH_ACCURACY is set, use 2x2 grid layout
HIGH_ACCURACY = os.getenv('HIGH_ACCURACY', '0')
if HIGH_ACCURACY in ('1', 'true', 'True', 'YES', 'yes'):
    # Use templates with 2x2 grid layout (4 equal squares per page)
    templates = ['333333333333']  # 12 panels in 3x4 grid  # Always 2x2 grid
    print("Using HIGH_ACCURACY mode with 2x2 grid layout (4 equal squares per page)")
else:
    # Optional grid layout for efficiency: when GRID_LAYOUT is set, prefer uniform grids
    GRID_LAYOUT = os.getenv('GRID_LAYOUT', '0')
    if GRID_LAYOUT in ('1', 'true', 'True', 'YES', 'yes'):
        # Use simple repetitive templates that create grid-like pages
        templates = ['6666', '4488', '44446', '666', '67']

# Adjust minimum length based on accuracy mode
if HIGH_ACCURACY in ('1', 'true', 'True', 'YES', 'yes'):
    min_length = 4  # Allow 4-image pages in high accuracy mode
else:
    min_length = 6
folder_path = 'frames/final' # Specify the folder path



def get_templates(input):
    page_templates = []
    start = 0

    while(start<len(input)):
        # print(f"start: {start}")
        result = []
        print(input)
        for template in templates:

            temp = input[start:start + len(template)]
            print(f"start: {start} len:{len(template)} temp:{temp}" )
            result.append(hammingDist(temp,template))            

       
        page_templates.append(templates[result.index(min(result))])

        start = start + len(templates[result.index(min(result))]) 



    if(len(temp) < min_length):
        if(len(temp) ==1):
          temp="5"
        elif(len(temp) ==2):
          temp="67"
        elif(len(temp) ==3):
          temp="666"
        elif(len(temp) ==4):
          temp="4488"
        elif(len(temp) ==5):
          temp="44446"

        page_templates[len(page_templates)-1] = temp
        # print("****************")

    return page_templates


def last_page(panels,count_images, length):
    count = 1
    
    # For high-accuracy mode, always use 2x2 grid
    if HIGH_ACCURACY in ('1', 'true', 'True', 'YES', 'yes'):
        if length == 1:
            # Single image takes full page
            new_panel = panel(f'frame{count_images:03d}', 4, 4)
            panels.append(new_panel)
        elif length == 2:
            # Two images: each takes half page
            new_panel = panel(f'frame{count_images:03d}', 2, 4)  # Top half
            panels.append(new_panel)
            count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 2, 4)  # Bottom half
            panels.append(new_panel)
        elif length == 3:
            # Three images: 2x2 grid with one panel empty
            new_panel = panel(f'frame{count_images:03d}', 2, 2)  # Top-left
            panels.append(new_panel)
            count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 2, 2)  # Top-right
            panels.append(new_panel)
            count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 2, 2)  # Bottom-left
            panels.append(new_panel)
        elif length == 4:
            # Perfect 2x2 grid
            for i in range(0, 4):
                new_panel = panel(f'frame{count_images:03d}', 2, 2)
                panels.append(new_panel)
                count_images += 1
        elif length == 5:
            # 2x2 grid plus one panel (will go to next page)
            for i in range(0, 4):
                new_panel = panel(f'frame{count_images:03d}', 2, 2)
                panels.append(new_panel)
                count_images += 1
    else:
        # Original logic for non-high-accuracy mode
        if length == 1:
            new_panel = panel(f'frame{count_images:03d}', 4, 4)  # Full page for single image
            panels.append(new_panel)
        elif length == 2:
            new_panel = panel(f'frame{count_images:03d}', 2, 4)  # Half page each
            panels.append(new_panel)
            count += 1
            count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 2, 4)
            panels.append(new_panel)
        elif length == 3:
            # 2x2 grid with one full-width panel
            new_panel = panel(f'frame{count_images:03d}', 2, 2)  # Top-left
            panels.append(new_panel)
            count += 1
            count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 2, 2)  # Top-right
            panels.append(new_panel)
            count += 1
            count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 4, 2)  # Bottom full-width
            panels.append(new_panel)
            count += 1
            count_images += 1
        elif length == 4:
            # Perfect 2x2 grid
            for i in range(0, 4):
                new_panel = panel(f'frame{count_images:03d}', 2, 2)
                panels.append(new_panel)
                count += 1
                count_images += 1
        elif length == 5:
            # 2x2 grid plus one full-width panel
            for i in range(0, 4):
                new_panel = panel(f'frame{count_images:03d}', 2, 2)
                panels.append(new_panel)
                count += 1
                count_images += 1
            new_panel = panel(f'frame{count_images:03d}', 4, 2)  # Full-width bottom
            panels.append(new_panel)
            count += 1
            count_images += 1

    return panels



def panel_create(page_templates):

    panels = []

    images = get_files_in_folder(folder_path)
    print(images)
    count_images = 1

    for page_template in page_templates:


        if(len(page_template)<min_length): #To handle last page 
            panels = last_page(panels,count_images,len(page_template))
            break


        count = 1
        
        # For high-accuracy mode, always create 2x2 grid
        if HIGH_ACCURACY in ('1', 'true', 'True', 'YES', 'yes'):
            # Create perfect 2x2 grid: each panel is 2x2
            for i in range(4):  # Always 4 panels per page
                new = panel(f'frame{count_images:03d}', 2, 2)  # 2 columns, 2 rows
                panels.append(new)
                count_images += 1
        else:
            # Original logic for non-high-accuracy mode
            for i in page_template:
                if(template_specs[i]['direction'] == 'row'):
                    new = panel(f'frame{count_images:03d}',template_specs[i]['span'] , 1)
                else:
                    new = panel(f'frame{count_images:03d}', 1 ,template_specs[i]['span'])
                panels.append(new)
                count = count+1
                count_images+=1

        
    
    return(panels)


# v = get_templates(input)
# print(v)
# new = panel_create(v)


# for i in new:
#     print(i.__dict__)