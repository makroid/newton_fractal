# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:51:17 2017
"""

import pyopencl as cl
import numpy as np

import os

os.environ['PYOPENCL_CTX'] = '0'


###############################################################################
## Helper classes


class FrclPar:
    def __init__(self):
        self.maxiter        = 32
        self.reverse        = 1
        self.cr             = 1.0
        self.ci             = 0.0
        self.delta_c        = 0.1
        self.roots_r        = None
        self.roots_i        = None
        self.fz_str         = ""
        self.dfz_str        = ""

    def incr_cr(self):
        self.cr += self.delta_c

    def decr_cr(self):
        self.cr -= self.delta_c

    def incr_ci(self):
        self.ci += self.delta_c

    def decr_ci(self):
        self.ci -= self.delta_c


class RandColoring:
    def __init__(self, stride, shift, idx):
        self.stride = stride
        self.shift = shift
        self.idx = idx


class ExportFile:
    def __init__(self):
        self.dir = None
        self.bn = "fractal_export"
        self.id = 0

    def get_next_filename(self):
        if self.dir is None:
            return None

        fn = os.path.join(self.dir, self.bn + "_" + str(self.id) + '.png')
        self.id += 1
        return fn


###############################################################################

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)

def fractal_opencl(q, frcl_pars):
    ## q is a list of complex coordinate values
    global ctx

    queue = cl.CommandQueue(ctx)

    output = np.zeros(q.shape, dtype=np.uint32)

    prg = cl.Program(ctx, """
    
    float2 plus(float2 z1, float2 z2) {
	    return (float2)(z1.x + z2.x, z1.y + z2.y);
	}
	
	float2 minus(float2 z1, float2 z2) {
	    return (float2)(z1.x - z2.x, z1.y - z2.y);
	}
	
	float2 mult(float2 z1, float2 z2) {
		return (float2)(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
	}
	float2 div(float2 z1, float2 z2) {
		if (z2.y == 0.0) {
			return (float2)(z2.x / z2.x, z1.y / z2.x);
    	}
    	if (z2.x == 0.0) {
    		return (float2)(z1.y / z2.y, -(z1.x / z2.y));
    	}
    	float r2 = z2.x * z2.x + z2.y * z2.y;
        return (float2)((z1.x * z2.x + z1.y * z2.y) / r2, (z1.y * z2.x - z1.x * z2.y) / r2);
	} 
	
	float2 complexExp(float2 z) {
		float expx = exp(z.x);
		return (float2)(expx*cos(z.y), expx*sin(z.y));
	}
	
	float2 complexLog(float2 z) {
		float r = sqrt(z.x*z.x + z.y*z.y);
		float theta = atan2(z.y, z.x);
			// if (theta < 0.0)
			//  theta += 2.0 * PI;
		return (float2)(log(r), theta);
	}
	
	float2 complexPower(float2 z, float2 a) {
		return complexExp(mult(a, complexLog(z)));
	}
    
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void newton_fractal(__global float2 *q,
                             __global uint *output,
                             __global float *roots_r,
                             __global float *roots_i,
                             ushort const n_roots, 
                             ushort const maxiter,
                             const float cr,
                             const float ci
                             )
    {
        int gid = get_global_id(0);
        float real = q[gid].x;
        float imag = q[gid].y;
        float2 z = (float2)(real, imag);
        output[gid] = 0;
        float tolerance = 0.000001f;
        
        for(int curiter = 0; curiter < maxiter; curiter++) {
            z = minus(z, mult((float2)(cr, ci), div(FUNC,DERIV)));
            
            for(int i=0; i<n_roots; i++) {
                float diff_r = z.x - roots_r[i];
                float diff_i = z.y - roots_i[i];
                
                if (fabs(diff_r) < tolerance && fabs(diff_i) < tolerance) {
                    output[gid] = i+1;
                    return;
                }
            }
        }
        output[gid] = 0;
    }
    """.replace("FUNC", frcl_pars.fz_str).replace("DERIV", frcl_pars.dfz_str)).build()

    mf              = cl.mem_flags
    q_opencl        = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    roots_r_opencl  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frcl_pars.roots_r)
    roots_i_opencl  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frcl_pars.roots_i)
    output_opencl   = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg.newton_fractal(queue, output.shape,
                None,
                q_opencl,
                output_opencl,
                roots_r_opencl,
                roots_i_opencl,
                np.uint16(len(frcl_pars.roots_r)),
                np.uint16(frcl_pars.maxiter),
                np.float32(frcl_pars.cr),
                np.float32(frcl_pars.ci))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output


def fractal_set(xmin, xmax, ymin, ymax, width, height, frcl_pars):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:, None] * 1j
    c = np.ravel(c)
    n3 = fractal_opencl(c, frcl_pars)
    n3 = n3.reshape((width, height))
    return (r1, r2, n3)


###############################################################################

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QLabel, QHBoxLayout, QFileDialog, QInputDialog
from PyQt5.QtGui import QImage, QPixmap, QTransform, QWheelEvent, QKeyEvent, QMouseEvent, QPainter, QBrush

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import sys


class ViewRect():
    def __init__(self):
        self._rect = QRectF(-1, -1, 2, 2)
        self._tmove = QTransform(1, 0, 0, 1, 0, 0)
        self._tview = QTransform(1, 0, 0, 1, 0, 0)
        self._taspect = QTransform(1, 0, 0, 1, 0, 0)

    def update_rect(self):
        trans = self._taspect * self._tmove * self._tview
        self.rect = trans.mapRect(QRectF(-1, -1, 2, 2))

    def map_px_to_coord(self, x, y, width, height):
        xs = (x - width/2) / (width/2)
        ys = (y - height/2) / (height/2)
        trans = self._taspect * self._tmove * self._tview
        trans_xy = trans.map(QPointF(xs, ys))
        return trans_xy

    def map_coord_to_px(self, x, y, width, height):
        fx = (x - self.min_x()) / (self.max_x() - self.min_x())
        fy = (y - self.min_y()) / (self.max_y() - self.min_y())
        px = int(fx * width)
        py = int(fy * height)
        return (px, py)

    def update_tview(self):
        self._tview = self._tmove * self._tview

    def update_translate(self, dx, dy):
        self._tmove = QTransform.fromTranslate(dx, dy)

    def reset_tmove(self):
        self._tmove = QTransform.fromTranslate(0, 0)

    def update_scale(self, sx, sy, px, py):
        self._tmove = QTransform.fromScale(sx, sy)

    def update_taspect(self, ratio):
        self._taspect = QTransform.fromScale(1, ratio)

    def set_scale(self, sx, sy, px, py):
        m = QTransform.fromTranslate(-px, -py)
        m = QTransform.fromScale(sx, sy) * m
        m = QTransform.fromTranslate(px, py) * m

        m = self._tmove * m
        self._tmove = m

    def min_x(self):
        return self.rect.left()

    def max_x(self):
        return self.rect.right()

    def min_y(self):
        return self.rect.top()

    def max_y(self):
        return self.rect.bottom()


class MPolynomial:
    def __init__(self):
        self.roots = [complex(-0.5, 0), complex(0.5, 0)]
        self.update()

    def update(self):
        if len(self.roots)>0:
            self.poly = Polynomial.fromroots(self.roots)
        else:
            self.poly = Polynomial([0])

    def add_root(self, root):
        self.roots.append(root)
        self.update()

    def remove_root(self, root_idx):
        assert root_idx >= 0 and root_idx < len(self.roots), "remove_root invalid index"
        del self.roots[root_idx]
        self.update()

    def get_roots(self):
        return self.roots

    def get_n_roots(self):
        return len(self.roots)

    def poly_as_str(self):
        prev = "(float2)(0.0f,0.0f)"
        for i,c in enumerate(self.poly.coef):
            formula = "plus(mult((float2)({}f,{}f), complexPower(z,(float2)({}.0f,0.0f))),{})".format(c.real, c.imag, i,
                                                                                                prev)
            prev = formula
        return prev

    def deriv_poly_as_str(self):
        deriv = self.poly.deriv()
        prev = "(float2)(0.0f,0.0f)"
        for i, c in enumerate(deriv.coef):
            formula = "plus(mult((float2)({}f,{}f), complexPower(z,(float2)({}.0f,0.0f))),{})".format(c.real, c.imag, i, prev)
            prev = formula
        return prev

###############################################################################
class NewtonFractalWidget(QWidget):

    def __init__(self, width, height, status_bar):
        super(NewtonFractalWidget, self).__init__()
        self.status_bar = status_bar

        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.imageLabel = None

        self.scale_fac_dwn      = 0.9
        self.scale_x            = 1
        self.scale_y            = 1
        self.rotate             = 0
        self.view_rect          = ViewRect()
        self.pos_down           = {'x': 0, 'y': 0}

        self.poly               = MPolynomial()
        self.draw_roots         = True

        self.export_file        = ExportFile()
        self.rand_coloring      = None

        self.coloring           = 1
        self.setup_colorTable()

        self.frcl_pars = FrclPar()

        self._init_imageLabel(width, height)

    def _init_imageLabel(self, width, height):
        if self.imageLabel is not None:
            self.layout.removeWidget(self.imageLabel)

        self.imageLabel = QLabel()
        self.imageLabel.setMinimumSize(1, 1)
        self.imageLabel.setMaximumSize(width, height)
        self.imageLabel.resize(width, height)

        self.layout.addWidget(self.imageLabel)
        self.update_fractal()

        self.resizeEvent = self.onResize

        self.imageLabel.installEventFilter(self)

    def update_fractal(self):
        self.update_status_bar()
        self.view_rect.update_rect()

        x_min = self.view_rect.min_x()
        x_max = self.view_rect.max_x()
        y_min = self.view_rect.min_y()
        y_max = self.view_rect.max_y()

        self.frcl_pars.roots_r = np.array([r.real for r in self.poly.roots], dtype=np.float32)
        self.frcl_pars.roots_i = np.array([r.imag for r in self.poly.roots], dtype=np.float32)

        self.frcl_pars.fz_str = self.poly.poly_as_str()
        self.frcl_pars.dfz_str = self.poly.deriv_poly_as_str()

        x, y, z = fractal_set(x_min, x_max, y_min, y_max, self.imageLabel.width(), self.imageLabel.height(),
                              self.frcl_pars)
        zz = z.ravel().astype(np.uint8)
        img = QImage(zz.data, self.imageLabel.width(), self.imageLabel.height(), self.imageLabel.width(),
                     QImage.Format_Indexed8)

        img.setColorTable(self.colorTable)

        self.imageLabel.setPixmap(QPixmap.fromImage(img))

        del img

    def check_existing_root(self, apx, apy):
        import math
        min = 99999999
        idx_min = -1
        for i, root in enumerate(self.poly.get_roots()):
            x, y = root.real, root.imag
            px, py = self.view_rect.map_coord_to_px(x, y, self.get_pixmap_w(), self.get_pixmap_h())
            dist = math.sqrt((px-apx)**2+(py-apy)**2)
            if dist < min:
                min = dist
                idx_min = i
        if min < 5:
            return idx_min
        else:
            return None

    def mousePressEvent(self, event):
        self.pos_down['x'] = event.x()
        self.pos_down['y'] = event.y()
        super(NewtonFractalWidget, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.update_fractal()
        self.pos_down['x'] = event.x()
        self.pos_down['y'] = event.y()

        self.view_rect.update_tview()
        self.view_rect.reset_tmove()

        if event.modifiers() & Qt.ControlModifier:
            idx_exist_root = self.check_existing_root(event.x(), event.y())

            if idx_exist_root is not None:
                self.poly.remove_root(idx_exist_root)
            else:
                multiple = 1
                if event.modifiers() & Qt.AltModifier:
                    multiple, okPressed = QInputDialog.getInt(self, "Multiplicity", "Multiplicity:", 3, 1, 20, 1)
                    if not okPressed:
                        multiple = 1

                p = self.view_rect.map_px_to_coord(event.x(), event.y(), self.get_pixmap_w(), self.get_pixmap_h())
                print("multiple=", multiple)
                for i in range(multiple):
                    root = complex(p.x(), p.y())
                    self.poly.add_root(root)
            self.setup_colorTable()
            self.update_fractal()
            self.repaint()

    def mouseMoveEvent(self, event):
        self.check_pan(event)
        super(NewtonFractalWidget, self).mousePressEvent(event)

    def get_pixmap_w(self):
        return self.imageLabel.pixmap().width()

    def get_pixmap_h(self):
        return self.imageLabel.pixmap().height()

    def keyPressEvent(self, e):
        self.check_pan(e)
        self.check_zoom(e)
        self.check_coloring(e)
        self.check_c(e)

        if e.key() == QtCore.Qt.Key_R:
            self.draw_roots = not self.draw_roots
            self.repaint()

        if e.modifiers() & Qt.ControlModifier and e.key() == QtCore.Qt.Key_S:
            self.save_view()

        if e.key() == QtCore.Qt.Key_S:
            self.save_view(mode="series")


    def save_view(self, mode="dialog"):
        if mode == "dialog":
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                       "All Files (*);;Text Files (*.txt)", options=options)
            if file_name:
                if file_name.endswith(('.png', '.jpg', '.jepg')):
                    self.imageLabel.pixmap().save(file_name)
                    self.export_file.dir = os.path.dirname(file_name)
                    self.export_file.id = 0
            self.update_status_bar("Save view to " + file_name)
        if mode == "series":
            file_name = self.export_file.get_next_filename()
            if file_name:
                self.imageLabel.pixmap().save(file_name)
                self.update_status_bar("Saved view to " + file_name)

    def paintEvent(self, event):
        if self.draw_roots:
            qp = QPainter()
            qp.begin(self.imageLabel.pixmap())
            qp.setRenderHint(QPainter.Antialiasing, True)
            qp.setBrush(QBrush(Qt.red, Qt.SolidPattern))

            for root in self.poly.get_roots():
                x, y = root.real, root.imag
                px, py = self.view_rect.map_coord_to_px(x, y, self.get_pixmap_w(), self.get_pixmap_h())
                qp.drawEllipse(QPointF(px, py), 7, 7)

            qp.end()

    def onResize(self, event):
        width, height = self.geometry().width(), self.geometry().height()
        ratio = (1.0 * height) / width
        self.view_rect.update_taspect(ratio)
        self._init_imageLabel(width, height)

    def check_pan(self, event):
        if isinstance(event, QMouseEvent):
            delta_x = (self.pos_down['x'] - event.x()) * 2 / self.imageLabel.width()
            delta_y = (self.pos_down['y'] - event.y()) * 2 / self.imageLabel.height()

            self.view_rect.update_translate(delta_x, delta_y)
            self.update_fractal()

        if isinstance(event, QKeyEvent):
            delta_x = 0
            delta_y = 0

            if event.key() == QtCore.Qt.Key_Left:
                delta_x = -0.1
            elif event.key() == QtCore.Qt.Key_Right:
                delta_x = 0.1
            elif event.key() == QtCore.Qt.Key_Up:
                delta_y = -0.1
            elif event.key() == QtCore.Qt.Key_Down:
                delta_y = 0.1

            self.view_rect.update_translate(delta_x, delta_y)
            self.update_fractal()

            self.view_rect.update_tview()
            self.view_rect.reset_tmove()

    def check_zoom(self, event):
        do_scale = False

        if isinstance(event, QKeyEvent):
            if event.key() == QtCore.Qt.Key_Plus:
                self.scale_x = self.scale_y = self.scale_fac_dwn
                do_scale = True
            if event.key() == QtCore.Qt.Key_Minus:
                self.scale_x = self.scale_y = 1 / self.scale_fac_dwn
                do_scale = True

        if isinstance(event, QWheelEvent):
            num_degs = event.angleDelta().y() / 8
            num_steps = num_degs / 15
            if num_steps > 0:
                self.scale_x = self.scale_y = self.scale_fac_dwn
            else:
                self.scale_x = self.scale_y = 1 / self.scale_fac_dwn
            do_scale = True

        if do_scale:
            dx = self.pos_down['x'] / self.imageLabel.width() * 2 - 1
            dy = self.pos_down['y'] / self.imageLabel.height() * 2 - 1

            self.view_rect.update_scale(self.scale_x, self.scale_y, dx, dy)
            self.view_rect.update_tview()
            self.view_rect.reset_tmove()
            self.update_fractal()

    def check_coloring(self, event):
        do_update = False

        if not isinstance(event, QKeyEvent):
            return

        if event.key() == QtCore.Qt.Key_1:
            self.coloring = 1
            do_update = True
        elif event.key() == QtCore.Qt.Key_2:
            self.coloring = 2
            do_update = True
        elif event.key() == QtCore.Qt.Key_3:
            self.coloring = 3
            do_update = True
        elif event.key() == QtCore.Qt.Key_4:
            self.coloring = 4
            do_update = True
        elif event.key() == QtCore.Qt.Key_5:
            self.coloring = 5
            do_update = True
        elif event.key() == QtCore.Qt.Key_6:
            self.coloring = 6
            do_update = True

        if do_update:
            self.setup_colorTable()
            self.update_fractal()

    def check_c(self, event):
        do_update = False

        if not isinstance(event, QKeyEvent):
            return

        if event.key() == QtCore.Qt.Key_F1:
            self.frcl_pars.decr_cr()
            do_update = True
        elif event.key() == QtCore.Qt.Key_F2:
            self.frcl_pars.incr_cr()
            do_update = True
        elif event.key() == QtCore.Qt.Key_F3:
            self.frcl_pars.decr_ci()
            do_update = True
        elif event.key() == QtCore.Qt.Key_F4:
            self.frcl_pars.incr_ci()
            do_update = True

        if do_update:
            self.update_fractal()

    def wheelEvent(self, event):
        self.check_zoom(event)

    def setup_colorTable(self):
        self.colorTable = []

        def get_spaced_colors(n):
            max_value = 16581375  # 255**3
            interval = int(max_value / n)
            colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

            return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

        if self.coloring == 1:
            colors = get_spaced_colors(self.poly.get_n_roots() + 1)
            self.colorTable = [QtGui.qRgb(r, g, b) for (r, g, b) in colors[1:]]
            #lines = g_colorings[0].split('\n')
            #for l in lines[1:]:
            #    r = int(l.split(",")[1])
            #    g = int(l.split(",")[2])
            #    b = int(l.split(",")[3])
            #    # rgb = (b << 0) | (g<<8) | (r<<16)
            #    self.colorTable.append(QtGui.qRgb(r, g, b))

        if self.coloring == 2:
            colors = get_spaced_colors(100 + 1)
            sample = np.random.choice(100, self.poly.get_n_roots(), replace=False)
            colors = [colors[i] for i in sample]
            self.colorTable = [QtGui.qRgb(r, g, b) for (r, g, b) in colors]

        elif self.coloring == 3:
            lines = g_colorings[1].split('\n')
            for l in lines[1:]:
                r = int(l.split(",")[1])
                g = int(l.split(",")[2])
                b = int(l.split(",")[3])
                # rgb = (b << 0) | (g<<8) | (r<<16)
                self.colorTable.append(QtGui.qRgb(r, g, b))

        elif self.coloring == 4:
            for r in range(255):
                self.colorTable.append(QtGui.qRgb(r, 0, 0))

        elif self.coloring == 5:
            for g in range(255):
                self.colorTable.append(QtGui.qRgb(0, g, 0))

        elif self.coloring == 6:
            ## sample a coloring
            stride = np.random.randint(1, 7)
            idx = np.random.choice(3, 2, replace=False)
            shift = np.random.randint(127)
            self.set_random_coloring(stride, shift, idx)
            self.rand_coloring = RandColoring(stride, shift, idx)


        elif self.coloring == 7:
            if self.rand_coloring is None:
                return

            self.set_random_coloring(self.rand_coloring.stride,
                                     self.rand_coloring.shift,
                                     self.rand_coloring.idx)

            self.rand_coloring.stride += 1
            self.rand_coloring.stride %= 100
            if self.rand_coloring.stride == 0:
                self.rand_coloring.stride += 1

        else:
            for b in range(128):
                if b % 3 == 0:
                    self.colorTable.append(QtGui.qRgb(0, b + 127, b + 127))
                    self.colorTable.append(QtGui.qRgb(0, b + 127, b + 127))
                else:
                    self.colorTable.append(QtGui.qRgb(0, 0, b + 127))
                    self.colorTable.append(QtGui.qRgb(0, 0, b + 127))

    def set_random_coloring(self, stride, shift, idx):
        rgb_1 = np.zeros((128, 3))
        rgb_2 = np.zeros((128, 3))

        for k in range(rgb_1.shape[0]):
            rgb_1[k, idx[0]] = k + shift
            rgb_1[k, idx[1]] = k + shift
            rgb_2[k, idx[1]] = k + shift

        for k in range(rgb_1.shape[0]):
            if k % stride == 0:
                self.colorTable.append(QtGui.qRgb(rgb_1[k, 0], rgb_1[k, 1], rgb_1[k, 2]))
                self.colorTable.append(QtGui.qRgb(rgb_1[k, 0], rgb_1[k, 1], rgb_1[k, 2]))
            else:
                self.colorTable.append(QtGui.qRgb(rgb_2[k, 0], rgb_2[k, 1], rgb_2[k, 2]))
                self.colorTable.append(QtGui.qRgb(rgb_2[k, 0], rgb_2[k, 1], rgb_2[k, 2]))

    def update_status_bar(self, msg_add=""):
        msg = "(re=" + str(self.frcl_pars.cr) + ", im=" + str(self.frcl_pars.ci) + ")" + " " + msg_add
        self.status_bar.showMessage(msg)


class FractalWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__()

        width = 500
        height = width

        self.setGeometry(100, 100, width, height)

        self.fractalWidget = NewtonFractalWidget(width, height, self.statusBar())
        self.fractalWidget.setFocusPolicy(Qt.StrongFocus)
        self.setFocusProxy(self.fractalWidget)
        self.fractalWidget.setFocus(True)

        self.setCentralWidget(self.fractalWidget)

        self.statusBar().showMessage('')

        self.show()


###############################################################################
## colorings form http://www.kennethmoreland.com/color-advice/

g_colorings = ["""
0.0,0,0,0
0.00392156862745,5,0,4
0.0078431372549,9,0,8
0.0117647058824,13,1,12
0.0156862745098,17,1,16
0.0196078431373,20,1,19
0.0235294117647,23,1,22
0.0274509803922,25,1,25
0.0313725490196,27,1,28
0.0352941176471,29,1,31
0.0392156862745,31,2,34
0.043137254902,32,2,37
0.0470588235294,34,2,40
0.0509803921569,35,2,43
0.0549019607843,36,2,46
0.0588235294118,37,2,49
0.0627450980392,38,2,52
0.0666666666667,39,3,54
0.0705882352941,40,3,57
0.0745098039216,41,3,60
0.078431372549,42,3,62
0.0823529411765,43,3,65
0.0862745098039,43,3,67
0.0901960784314,44,3,70
0.0941176470588,45,3,72
0.0980392156863,45,4,75
0.101960784314,46,4,78
0.105882352941,47,4,80
0.109803921569,47,4,83
0.113725490196,48,4,86
0.117647058824,49,4,89
0.121568627451,49,4,92
0.125490196078,50,5,94
0.129411764706,51,5,97
0.133333333333,51,5,100
0.137254901961,52,5,103
0.141176470588,52,5,106
0.145098039216,53,5,109
0.149019607843,53,5,111
0.152941176471,54,5,114
0.156862745098,55,6,117
0.160784313725,55,6,120
0.164705882353,56,6,122
0.16862745098,57,6,125
0.172549019608,57,6,128
0.176470588235,58,6,130
0.180392156863,59,6,133
0.18431372549,60,6,135
0.188235294118,61,7,138
0.192156862745,61,7,140
0.196078431373,62,7,143
0.2,63,7,145
0.203921568627,64,7,148
0.207843137255,64,7,151
0.211764705882,64,7,155
0.21568627451,64,8,158
0.219607843137,64,8,162
0.223529411765,63,8,166
0.227450980392,62,8,171
0.23137254902,60,8,176
0.235294117647,57,9,182
0.239215686275,53,9,189
0.243137254902,47,9,196
0.247058823529,37,10,205
0.250980392157,19,12,213
0.254901960784,13,20,211
0.258823529412,10,27,208
0.262745098039,10,33,204
0.266666666667,10,38,200
0.270588235294,9,43,196
0.274509803922,9,47,191
0.278431372549,9,50,187
0.282352941176,9,54,183
0.286274509804,9,57,179
0.290196078431,8,60,175
0.294117647059,8,62,171
0.298039215686,8,65,167
0.301960784314,8,67,163
0.305882352941,8,69,159
0.309803921569,8,72,156
0.313725490196,7,74,152
0.317647058824,7,75,149
0.321568627451,7,77,146
0.325490196078,7,79,143
0.329411764706,7,81,140
0.333333333333,7,83,137
0.337254901961,7,84,134
0.341176470588,6,86,132
0.345098039216,6,87,129
0.349019607843,6,89,127
0.352941176471,6,90,125
0.356862745098,6,92,123
0.360784313725,6,93,121
0.364705882353,6,94,119
0.36862745098,6,96,117
0.372549019608,6,97,116
0.376470588235,5,98,114
0.380392156863,5,100,112
0.38431372549,5,101,111
0.388235294118,5,102,110
0.392156862745,5,104,108
0.396078431373,5,105,107
0.4,5,106,106
0.403921568627,5,107,105
0.407843137255,5,109,104
0.411764705882,5,110,102
0.41568627451,5,111,101
0.419607843137,5,112,100
0.423529411765,5,113,99
0.427450980392,5,115,97
0.43137254902,6,116,96
0.435294117647,6,117,95
0.439215686275,6,118,93
0.443137254902,6,120,92
0.447058823529,6,121,91
0.450980392157,6,122,89
0.454901960784,6,123,88
0.458823529412,6,124,86
0.462745098039,6,126,85
0.466666666667,6,127,83
0.470588235294,6,128,81
0.474509803922,6,129,80
0.478431372549,6,131,78
0.482352941176,6,132,77
0.486274509804,7,133,75
0.490196078431,6,134,73
0.494117647059,7,135,71
0.498039215686,7,137,70
0.501960784314,7,138,68
0.505882352941,7,139,66
0.509803921569,7,140,64
0.513725490196,7,142,62
0.517647058824,7,143,60
0.521568627451,7,144,58
0.525490196078,7,145,56
0.529411764706,7,146,54
0.533333333333,7,148,52
0.537254901961,7,149,50
0.541176470588,7,150,48
0.545098039216,7,151,46
0.549019607843,7,152,44
0.552941176471,7,154,42
0.556862745098,7,155,41
0.560784313725,8,156,39
0.564705882353,8,157,37
0.56862745098,8,158,35
0.572549019608,8,160,34
0.576470588235,8,161,32
0.580392156863,8,162,31
0.58431372549,8,163,30
0.588235294118,8,164,29
0.592156862745,8,165,28
0.596078431373,8,167,27
0.6,8,168,26
0.603921568627,8,169,26
0.607843137255,8,170,26
0.611764705882,8,171,25
0.61568627451,8,173,25
0.619607843137,8,174,24
0.623529411765,9,175,23
0.627450980392,9,176,22
0.63137254902,9,177,20
0.635294117647,9,179,18
0.639215686275,9,180,14
0.643137254902,9,181,10
0.647058823529,15,182,9
0.650980392157,22,183,9
0.654901960784,29,184,9
0.658823529412,34,185,9
0.662745098039,40,186,9
0.666666666667,45,187,9
0.670588235294,50,188,9
0.674509803922,54,189,9
0.678431372549,59,190,9
0.682352941176,64,191,9
0.686274509804,69,191,9
0.690196078431,73,192,9
0.694117647059,78,193,9
0.698039215686,82,194,9
0.701960784314,87,195,9
0.705882352941,91,195,9
0.709803921569,96,196,9
0.713725490196,100,197,9
0.717647058824,105,197,10
0.721568627451,109,198,10
0.725490196078,113,199,9
0.729411764706,118,199,10
0.733333333333,122,200,10
0.737254901961,127,200,10
0.741176470588,131,201,10
0.745098039216,135,201,10
0.749019607843,140,202,10
0.752941176471,144,202,10
0.756862745098,149,203,10
0.760784313725,153,203,10
0.764705882353,157,204,10
0.76862745098,162,204,10
0.772549019608,166,204,10
0.776470588235,170,205,10
0.780392156863,175,205,10
0.78431372549,179,205,10
0.788235294118,183,206,10
0.792156862745,187,206,10
0.796078431373,192,206,10
0.8,196,206,10
0.803921568627,200,206,10
0.807843137255,205,207,10
0.811764705882,209,207,10
0.81568627451,214,207,10
0.819607843137,218,207,10
0.823529411765,223,206,11
0.827450980392,228,206,11
0.83137254902,233,206,11
0.835294117647,238,206,11
0.839215686275,244,205,17
0.843137254902,246,205,64
0.847058823529,247,206,91
0.850980392157,248,206,109
0.854901960784,249,207,123
0.858823529412,249,208,135
0.862745098039,250,209,144
0.866666666667,250,210,153
0.870588235294,250,211,160
0.874509803922,251,212,167
0.878431372549,251,213,173
0.882352941176,251,214,178
0.886274509804,252,215,183
0.890196078431,252,217,187
0.894117647059,252,218,191
0.898039215686,252,219,195
0.901960784314,252,220,199
0.905882352941,252,222,202
0.909803921569,253,223,205
0.913725490196,253,224,208
0.917647058824,253,226,211
0.921568627451,253,227,214
0.925490196078,253,228,217
0.929411764706,253,230,219
0.933333333333,253,231,222
0.937254901961,254,233,224
0.941176470588,254,234,226
0.945098039216,254,235,228
0.949019607843,254,237,231
0.952941176471,254,238,233
0.956862745098,254,240,235
0.960784313725,254,241,237
0.964705882353,254,242,239
0.96862745098,254,244,240
0.972549019608,254,245,242
0.976470588235,254,247,244
0.980392156863,255,248,246
0.98431372549,255,249,248
0.988235294118,255,251,250
0.992156862745,255,252,251
0.996078431373,255,254,253
1.0,255,255,255""",
               """
               0.0,0,0,0
               0.00787401574803,9,0,8
               0.0157480314961,17,1,16
               0.0236220472441,23,1,22
               0.0314960629921,27,1,28
               0.0393700787402,31,2,34
               0.0472440944882,34,2,40
               0.0551181102362,36,2,46
               0.0629921259843,38,2,52
               0.0708661417323,40,3,57
               0.0787401574803,42,3,62
               0.0866141732283,43,3,67
               0.0944881889764,45,3,73
               0.102362204724,46,4,78
               0.110236220472,48,4,84
               0.11811023622,49,4,89
               0.125984251969,50,5,95
               0.133858267717,51,5,100
               0.141732283465,52,5,106
               0.149606299213,54,5,112
               0.157480314961,55,6,117
               0.165354330709,56,6,123
               0.173228346457,58,6,128
               0.181102362205,59,6,133
               0.188976377953,61,7,138
               0.196850393701,63,7,143
               0.204724409449,64,7,149
               0.212598425197,64,7,155
               0.220472440945,64,8,163
               0.228346456693,62,8,172
               0.236220472441,57,9,184
               0.244094488189,45,9,198
               0.251968503937,18,14,213
               0.259842519685,10,29,207
               0.267716535433,10,40,199
               0.275590551181,9,48,190
               0.283464566929,9,55,182
               0.291338582677,8,60,173
               0.299212598425,8,66,166
               0.307086614173,8,70,158
               0.314960629921,7,74,151
               0.322834645669,7,78,145
               0.330708661417,7,81,139
               0.338582677165,6,85,134
               0.346456692913,6,88,129
               0.354330708661,6,91,124
               0.362204724409,6,94,120
               0.370078740157,6,96,117
               0.377952755906,6,99,113
               0.385826771654,5,102,111
               0.393700787402,5,104,108
               0.40157480315,5,107,105
               0.409448818898,5,109,103
               0.417322834646,5,112,101
               0.425196850394,5,114,98
               0.433070866142,6,116,95
               0.44094488189,6,119,93
               0.448818897638,6,121,90
               0.456692913386,6,124,87
               0.464566929134,6,126,84
               0.472440944882,6,129,81
               0.48031496063,6,131,77
               0.488188976378,7,134,74
               0.496062992126,7,136,71
               0.503937007874,7,139,67
               0.511811023622,7,141,63
               0.51968503937,7,143,59
               0.527559055118,7,146,55
               0.535433070866,7,148,51
               0.543307086614,7,151,47
               0.551181102362,8,153,43
               0.55905511811,8,155,39
               0.566929133858,8,158,36
               0.574803149606,8,160,33
               0.0,0,0,0
               0.00787401574803,9,0,8
               0.0157480314961,17,1,17
               0.0236220472441,22,1,23
               0.0314960629921,27,1,30
               0.0393700787402,30,2,37
               0.0472440944882,32,2,44
               0.0551181102362,34,2,52
               0.0629921259843,35,3,59
               0.0708661417323,36,3,65
               0.0787401574803,37,3,71
               0.0866141732283,38,4,77
               0.0944881889764,39,4,83
               0.102362204724,40,4,89
               0.110236220472,41,4,94
               0.11811023622,42,5,99
               0.125984251969,44,5,104
               0.133858267717,44,5,110
               0.141732283465,43,6,118
               0.149606299213,36,6,131
               0.157480314961,10,7,149
               0.165354330709,7,22,140
               0.173228346457,6,31,129
               0.181102362205,6,38,119
               0.188976377953,5,43,109
               0.196850393701,5,48,101
               0.204724409449,4,51,93
               0.212598425197,4,55,87
               0.220472440945,4,58,81
               0.228346456693,4,60,77
               0.236220472441,3,63,73
               0.244094488189,3,65,69
               0.251968503937,3,68,66
               0.259842519685,3,70,64
               0.267716535433,4,72,61
               0.275590551181,4,75,57
               0.283464566929,4,77,54
               0.291338582677,4,79,51
               0.299212598425,4,82,47
               0.307086614173,4,84,43
               0.314960629921,4,86,40
               0.322834645669,4,88,36
               0.330708661417,4,91,31
               0.338582677165,5,93,27
               0.346456692913,5,95,23
               0.354330708661,5,97,19
               0.362204724409,5,100,16
               0.370078740157,5,102,14
               0.377952755906,5,104,13
               0.385826771654,5,106,12
               0.393700787402,5,108,10
               0.40157480315,6,111,5
               0.409448818898,17,112,5
               0.417322834646,26,114,5
               0.425196850394,36,116,6
               0.433070866142,44,117,6
               0.44094488189,53,118,6
               0.448818897638,62,120,6
               0.456692913386,71,121,6
               0.464566929134,79,122,6
               0.472440944882,88,122,6
               0.48031496063,97,123,6
               0.488188976378,105,123,6
               0.496062992126,113,124,6
               0.503937007874,122,124,6
               0.511811023622,131,124,6
               0.51968503937,140,123,7
               0.527559055118,151,122,7
               0.535433070866,162,120,8
               0.543307086614,174,118,8
               0.551181102362,186,114,9
               0.55905511811,199,109,10
               0.566929133858,213,103,10
               0.574803149606,227,96,11
               0.582677165354,241,86,11
               0.590551181102,244,87,33
               0.59842519685,245,91,48
               0.606299212598,246,95,58
               0.614173228346,246,99,66
               0.622047244094,246,103,72
               0.629921259843,247,107,79
               0.637795275591,247,110,89
               0.645669291339,248,113,100
               0.653543307087,248,116,112
               0.661417322835,249,119,125
               0.669291338583,249,122,138
               0.677165354331,249,125,150
               0.685039370079,249,128,162
               0.692913385827,249,130,173
               0.700787401575,249,133,184
               0.708661417323,249,136,195
               0.716535433071,249,138,205
               0.724409448819,250,141,214
               0.732283464567,250,143,223
               0.740157480315,250,146,232
               0.748031496063,250,149,239
               0.755905511811,250,151,247
               0.763779527559,247,157,250
               0.771653543307,241,163,251
               0.779527559055,237,169,251
               0.787401574803,234,174,251
               0.795275590551,231,179,251
               0.803149606299,229,183,252
               0.811023622047,228,187,252
               0.818897637795,227,190,252
               0.826771653543,227,194,252
               0.834645669291,227,197,252
               0.842519685039,228,200,252
               0.850393700787,228,203,253
               0.858267716535,229,206,253
               0.866141732283,231,209,253
               0.874015748031,232,211,253
               0.88188976378,233,214,253
               0.889763779528,234,217,253"""
               ]

###############################################################################


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FractalWindow()
    sys.exit(app.exec_())