import csv
import os
import re
import warnings
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

import Image2CAD


class FileHandle:
    """
    Handles rendering of images from PDF and checking file type.
    Parameters: filename - str
                folder_path - str
    """

    def __init__(self, filename, folder_path):
        self.path = folder_path + filename
        self.filename = filename
        self.filetype = self.filename.lower().split('.')[1]
        self.image = None
        self.get_image()

    def get_image(self):
        if self.filetype == 'pdf':
            pdf = fitz.open(self.path)

            if pdf.page_count > 1:
                raise Exception(
                    "PDF: {fname} has more than 1 pages. Remove extra pages and try again".format(fname=self.filename))

            page = pdf.loadPage(0)
            pix = page.getPixmap(matrix=fitz.Matrix(300 / 75, 300 / 75), alpha=False)  # 300DPI images
            # pil_image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            print('Image rendered from PDF')
            self.image = self.pix2np(pix)


        elif self.filetype == 'jpeg' or self.filetype == 'jpg' or self.filetype == 'png':
            self.image = cv2.imread(self.path)

        else:
            raise Exception('Unknown file extension: {fname}'.format(fname=self.filetype))
        # im_pil = Image.fromarray(self.cv_image)
        # im_pil.show()

    def pix2np(self, pix):
        im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
        return im


class DrawingProcess:
    def __init__(self, drawing_image):
        self.drawing_image = drawing_image
        self.resolution = [self.drawing_image.shape[1], self.drawing_image.shape[0]]

        self.gray = cv2.cvtColor(self.drawing_image, cv2.COLOR_BGR2GRAY)
        self.no_border = None  # Drawing image with no borders
        self.border_rect = []
        self.diagram_rects = []
        self.diagrams = []
        self.get_drawings()

        self.largest_x = 0
        self.largest_y = 0
        self.largest_z = 0

    def get_drawings(self):
        self.no_border = self.crop_rect(self.get_border())

        gray = cv2.cvtColor(self.no_border, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        diagram_contours = []
        for c in cnts:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            size_factor = 0.2
            # if 0.5 * self.resolution[0] > w > size_factor * self.resolution[0] or 0.5 * self.resolution[
            #     1] > h > size_factor * self.resolution[1]:
            if w < 0.05 * self.resolution[0] or h < 0.05 * self.resolution[1]:
                continue
            elif w > 0.9 * self.resolution[0] or h > 0.9 * self.resolution[1]:
                continue
            diagram_contours.append(c)

        _, self.diagram_rects = self.sort_contours(diagram_contours)

        print('total rects', len(self.diagram_rects))

    @staticmethod
    def sort_contours(cnts, method="top-to-bottom"):
        rect = [cv2.boundingRect(i) for i in cnts]
        c  = np.array(rect)
        max_height = np.max(c[::, 3])
        nearest = max_height * 1

        rect.sort(key=lambda r: [int(nearest * round(float(r[1]) / nearest)), r[0]])
        return (cnts, rect)


    def get_border(self):
        gray_image = self.gray

        blur = cv2.GaussianBlur(gray_image, (31, 31), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(cnts, key=cv2.contourArea, reverse=True)

        rect = cv2.minAreaRect(contours[0])
        return rect

    def crop_rect(self, rect, margin_cut=None):
        margin_cut = 0.05 * self.resolution[0]
        # rotate img
        img = self.drawing_image
        rect = list(rect)
        rect[1] = list(rect[1])
        rect[1] = (rect[1][0] - margin_cut, rect[1][1] - margin_cut)
        self.border_rect = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))

        if rect[2] > 45:
            warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)
        return warped

    def draw_diagram_borders(self):
        show_image = self.no_border.copy()
        num = 1
        for r in self.diagram_rects:
            x, y, w, h = r
            num_pos_x = x+w//2
            num_pos_y = y + h//2
            cv2.putText(show_image, str(num), (num_pos_x,num_pos_y), cv2.FONT_HERSHEY_SIMPLEX,
                        10, (255, 0, 0), 6, cv2.LINE_AA)
            cv2.rectangle(show_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            num+=1
        pil_show(show_image)


    def start_diagram_processing(self):
        for rect in self.diagram_rects:
            x, y, w, h = rect
            image = self.no_border[y:y + h, x:x + w].copy()
            self.diagrams.append(DiagramProcess(image, drawing_resolution=self.resolution))

        self.diagrams[0].get_text()
        self.diagrams[0].get_arrows_and_dimension_lines()
        self.largest_x, self.largest_y = self.diagrams[0].analyze_data()

        self.diagrams[1].get_text()
        self.diagrams[1].get_arrows_and_dimension_lines()
        z_1, z_2 = self.diagrams[1].analyze_data()
        self.largest_z = [z_1, z_2]


class DiagramProcess:
    def __init__(self, image, drawing_resolution):
        self.image = image
        self.drawing_resolution = drawing_resolution

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.scaling_for_arrows = None  # Scaling Value required for proper arrow detection
        self.text_data = []
        self.arrow_data = []

    def remove_lines(self, image):
        kernal_dim = int(self.drawing_resolution[0] * 0.01)

        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_dim, 1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernal_dim))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(result, [c], -1, (255, 255, 255), 5)
        return result

    def get_text(self):
        scaling_buffer = []  # Contains possible values of scaling values... at the end the median will be chosen
        drawing_with_lines = self.image.copy()
        drawing = self.remove_lines(drawing_with_lines)
        gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print(len(cnts))
        regex = re.compile('[R0123456789x=.,]')
        for c in cnts:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            w += 10
            h += 10
            box_now = drawing_with_lines[y:y + h, x:x + w]

            if w > 0.2 * drawing_with_lines.shape[1]:
                continue
            if h > w:
                box_now = cv2.rotate(box_now, cv2.cv2.ROTATE_90_CLOCKWISE)

            ocr_data = pytesseract.image_to_data(box_now,
                                                 lang='eng', output_type=pytesseract.Output.DICT,
                                                 config='--psm 7 -c tessedit_char_whitelist=R0123456789x=.,')
            ocr_text = ''.join(ocr_data['text'])
            ocr_text = ocr_text.replace('\n\x0c', '')  # Removes new line and form feed character

            # print('re test', regex.search(ocr_text))
            if (regex.search(ocr_text) == None):
                continue

            n_boxes = len(ocr_data['level'])
            for i in range(n_boxes):
                if ocr_data['text'][i] != '':
                    if w > h:
                        alignment = 'Horizontal'
                        (x_b, y_b, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], # TODO: Make this more elegant
                                            ocr_data['height'][i])
                    else:
                        (x_b, y_b, h, w) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i],
                                            ocr_data['height'][i])
                        alignment = 'Vertical'
            if ocr_text.isnumeric() and self.scaling_for_arrows is None:
                digit_len = 14
                no_digits = len(ocr_text)
                total_length = max(w, h)
                # res_attrib = drawing_with_lines.shape[1]
                foo = total_length / (no_digits)
                scale_value = digit_len / foo
                scaling_buffer.append(scale_value)

            # print(ocr_text)
            x = x + x_b
            y = y + y_b
            self.text_data.append([ocr_text,(x,y),(w,h),alignment])
            cv2.rectangle(drawing, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.putText(drawing, ocr_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        pil_show(drawing)

        scaling_buffer.sort()
        self.scaling_for_arrows = scaling_buffer[len(scaling_buffer) // 2]
        print('Scaling for arrows: ', self.scaling_for_arrows)

    def get_arrows_and_dimension_lines(self, scale = None):

        drawing_with_lines = self.image

        if scale is None:
            scale = self.scaling_for_arrows
        print('scale: ', scale)
        if scale < 1:
            interpol = cv2.INTER_AREA
        else:
            interpol = cv2.INTER_CUBIC
        drawing_with_lines = cv2.resize(drawing_with_lines, None, fx=scale, fy=scale, interpolation=interpol)
        # pil_show(drawing_with_lines)
        cv2.imwrite('temp.png', drawing_with_lines)
        self.arrow_data, self.segment_lines, self.erased_img  = Image2CAD.main('temp.png',scale, show=False)

        for arrow in self.arrow_data:
            if arrow["Direction"] == '':
                image = cv2.circle(self.image, (int(arrow['x']), int(arrow['y']) + 9), 5, (255, 0, 0), -1)
            else:
                image = cv2.circle(self.image, (int(arrow['x']), int(arrow['y']) + 9), 5, (0, 0, 255), -1)
        for line in self.segment_lines:
            if int(line[0][0])==int(111.21):
                print('color changed')
                color = (0,0,255)
            else:
                color = (255,0,0)
            start = tuple(map(int, line[0]))
            end = tuple(map(int, line[1]))
            image = cv2.line(self.image, start, end, color, 3)

        # pil_show(image)
        # pil_show(self.erased_img)
        # cv2.imwrite(r"D:\My Projects\Python Projects\Engineering Drawing OCR\Drawings\Images\fill_test.png",self.erased_img)

    def floodfill_segment(self):
        drawing_with_lines = self.image.copy()

        pil_show(self.erased_img)
        image = cv2.cvtColor(self.erased_img, cv2.COLOR_BGR2GRAY)
        # mask = np.zeros(drawing_with_lines.shape[:-1], np.uint8)
        cv2.floodFill(image, None, (0,0), 0)
        pil_show(image)

    def analyze_data(self):
        largest_horiz = 0
        largest_vertical = 0
        for data in self.text_data:
            text = data[0]
            alignment = data[3]
            if text.isdigit():
                if alignment == 'Horizontal' and int(text)>largest_horiz:
                    largest_horiz = int(text)
                elif   alignment == 'Vertical' and int(text)>largest_vertical:
                    largest_vertical = int(text)
        print('large before',largest_vertical,largest_horiz)


        grouped_segments = self.find_collinear_lines(self.segment_lines)

        # test_line = grouped_segments[0][0]
        # print('this is found text',self.find_text(*test_line))
        for segment in grouped_segments:
            sum = 0
            for line in segment:
                text = self.find_text(*line)
                print(text)
                if text.isdigit():
                    sum+=int(text)
            if line[2] == 'Horizontal' and sum>largest_horiz:
                largest_horiz = sum
            if line[2] == 'Vertical' and sum>largest_vertical:
                largest_vertical = sum

        return largest_horiz, largest_vertical

    def find_text(self,start, end, alignment):

        if alignment == 'Horizontal':
            index_1 = 0
            index_2 = 1
            offset = -50
        elif alignment == 'Vertical':
            index_1 = 1
            index_2 = 0
            offset = -50
        else:
            return '0'
            raise Warning('Slanted Lines found')

        for data in self.text_data:
            pos = data[1]

            if start[index_1]<=pos[index_1]<=end[index_1] and start[index_2]+offset<pos[index_2]<start[index_2]:
                return data[0]

        return '0'

    @staticmethod
    def find_collinear_lines(segments): #TODO: Nuke this annd write it better ;(
        segment_copy = segments.copy()
        grouped_segments = []
        num = 0
        while len(segment_copy) > 0:
            line_now = segment_copy[0]
            align = line_now[2]
            index_now = segments.index(line_now)
            grouped_segments.append([line_now])
            segment_copy.remove(line_now)
            for index, line in enumerate(segments):
                if index == index_now:
                    continue

                if line[2] == align:
                    if align == 'Horizontal' and line[0][1] == line_now[0][1]:
                        grouped_segments[num].append(line)
                        segment_copy.remove(line)
                    elif align == 'Vertical' and line[0][0] == line_now[0][0]:
                        grouped_segments[num].append(line)
                        segment_copy.remove(line)
            num+=1
        return grouped_segments

def pil_show(img):
    im_pil = Image.fromarray(img)

    # im_pil.show()

if __name__ == "__main__":
    # PB72692A.PDF 279721107.pdf PB74253A.PDF PC65005B.PDF

    # drawing = FileHandle('PB74253A.PDF', 'Drawings\\')
    # drawing = FileHandle('279721107.pdf', 'Drawings\\')
    # drawing_process = DrawingProcess(drawing.image)
    # drawing_process.draw_diagram_borders()
    # drawing_process.start_diagram_processing()
    # print(drawing_process.largest_x, drawing_process.largest_y, drawing_process.largest_z)

    # TODO: Work all these into the fileHandle class
    csv_filename = "found_dimensions.csv"
    file_exist = Path(csv_filename).exists()
    csv_file = open(csv_filename, 'a')
    headers = ['Filename', 'Found Value - x', 'Found Value - y', 'Found Value(s) - z', 'File Path']
    writer = csv.writer(csv_file)
    reader = csv.reader(csv_file)

    if not file_exist:
        writer.writerow(headers)

    pc_location = "D:\My Projects\Python Projects\Engineering Drawing OCR\Drawings\\"
    drawings_path = 'Drawings'
    drawing_files = [f for f in os.listdir(drawings_path)
                     if os.path.isfile(os.path.join(drawings_path, f))]  # Gets only the files (not directories)

    for image_file in drawing_files:
        try:
            drawing = FileHandle(image_file, 'Drawings\\')
            drawing_process = DrawingProcess(drawing.image)
            drawing_process.draw_diagram_borders()
            drawing_process.start_diagram_processing()
            print(drawing_process.largest_x, drawing_process.largest_y, drawing_process.largest_z)
            row = [image_file, drawing_process.largest_x, drawing_process.largest_y, drawing_process.largest_z,
                   os.path.join(pc_location, image_file)]
            writer.writerow(row)
        except Exception as e:
            file_error = "Error with the following file: {fname}".format(fname=image_file)
            warnings.warn(file_error)
            print('-------------------------------------------------------')
            print(e)
            print('-------------------------------------------------------')
            continue
