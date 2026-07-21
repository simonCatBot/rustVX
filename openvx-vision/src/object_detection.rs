//! Object detection operations

use openvx_core::{Context, KernelTrait, Referenceable, VxKernel, VxResult};
use openvx_image::Image;

/// CannyEdgeDetector kernel
pub struct CannyEdgeDetectorKernel;

impl CannyEdgeDetectorKernel {
    pub fn new() -> Self {
        CannyEdgeDetectorKernel
    }
}

impl KernelTrait for CannyEdgeDetectorKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.canny_edge_detector"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::CannyEdgeDetector
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(1)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // Default thresholds
        let low_threshold = 50u8;
        let high_threshold = 150u8;

        canny_edge_detector(src, dst, low_threshold, high_threshold)?;
        Ok(())
    }
}

/// HoughLinesP kernel - probabilistic Hough transform
pub struct HoughLinesPKernel;

impl HoughLinesPKernel {
    pub fn new() -> Self {
        HoughLinesPKernel
    }
}

impl KernelTrait for HoughLinesPKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.hough_lines_p"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::HoughLinesP
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 2 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        // Default parameters
        let rho = 1.0f32;
        let theta = std::f32::consts::PI / 180.0;
        let threshold = 50;
        let min_line_length = 10;
        let max_line_gap = 10;

        let _lines = hough_lines_p(src, rho, theta, threshold, min_line_length, max_line_gap)?;

        Ok(())
    }
}

/// Threshold kernel - apply threshold to image
/// Supports BINARY: dst = (src > thresh) ? maxval : 0
/// Supports RANGE: dst = (src >= lower && src <= upper) ? maxval : 0
pub struct ThresholdKernel {
    pub thresh_type: ThresholdType,
    pub lower: u8,
    pub upper: u8,
    pub maxval: u8,
}

impl ThresholdKernel {
    pub fn new() -> Self {
        ThresholdKernel {
            thresh_type: ThresholdType::Binary,
            lower: 128,
            upper: 255,
            maxval: 255,
        }
    }

    pub fn with_binary(thresh: u8, maxval: u8) -> Self {
        ThresholdKernel {
            thresh_type: ThresholdType::Binary,
            lower: thresh,
            upper: 255,
            maxval,
        }
    }

    pub fn with_range(lower: u8, upper: u8, maxval: u8) -> Self {
        ThresholdKernel {
            thresh_type: ThresholdType::Range,
            lower,
            upper,
            maxval,
        }
    }
}

impl KernelTrait for ThresholdKernel {
    fn get_name(&self) -> &str {
        "org.khronos.openvx.threshold"
    }
    fn get_enum(&self) -> VxKernel {
        VxKernel::Threshold
    }

    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()> {
        if params.len() < 3 {
            return Err(openvx_core::VxStatus::ErrorInvalidParameters);
        }
        Ok(())
    }

    fn execute(&self, params: &[&dyn Referenceable], _context: &Context) -> VxResult<()> {
        let src = params
            .get(0)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;
        let dst = params
            .get(2)
            .and_then(|p| p.as_any().downcast_ref::<Image>())
            .ok_or(openvx_core::VxStatus::ErrorInvalidParameters)?;

        match self.thresh_type {
            ThresholdType::Binary => {
                threshold_binary(src, dst, self.lower, self.maxval)?;
            }
            ThresholdType::Range => {
                threshold_range(src, dst, self.lower, self.upper, self.maxval)?;
            }
        }
        Ok(())
    }
}

/// Line segment structure
#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl LineSegment {
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        LineSegment { x1, y1, x2, y2 }
    }

    /// Compute line length
    pub fn length(&self) -> f32 {
        let dx = (self.x2 - self.x1) as f32;
        let dy = (self.y2 - self.y1) as f32;
        (dx * dx + dy * dy).sqrt()
    }

    /// Compute line angle in radians
    pub fn angle(&self) -> f32 {
        let dy = (self.y2 - self.y1) as f32;
        let dx = (self.x2 - self.x1) as f32;
        dy.atan2(dx)
    }
}

/// Canny edge detector
/// Steps:
/// 1. Apply Gaussian blur
/// 2. Compute gradient magnitude and direction
/// 3. Non-maximum suppression
/// 4. Double threshold
/// 5. Edge tracking by hysteresis
pub fn canny_edge_detector(
    src: &Image,
    dst: &Image,
    low_threshold: u8,
    high_threshold: u8,
) -> VxResult<()> {
    let width = src.width();
    let height = src.height();

    // Step 1: Apply Gaussian blur (simplified - use existing 3x3)
    // Use saturating_mul to prevent integer overflow
    let image_size = width.saturating_mul(height);
    let mut blurred = vec![0u8; image_size];
    {
        let kernel = [1, 2, 1];
        let mut temp = vec![0u8; image_size];

        // Horizontal pass
        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..3 {
                    let px = x as isize + k as isize - 1;
                    if px >= 0 && px < width as isize {
                        sum += src.get_pixel(px as usize, y) as i32 * kernel[k];
                        weight += kernel[k];
                    }
                }
                temp[y * width + x] = (sum / weight.max(1)) as u8;
            }
        }

        // Vertical pass
        for y in 0..height {
            for x in 0..width {
                let mut sum: i32 = 0;
                let mut weight: i32 = 0;
                for k in 0..3 {
                    let py = y as isize + k as isize - 1;
                    if py >= 0 && py < height as isize {
                        sum += temp[py as usize * width + x] as i32 * kernel[k];
                        weight += kernel[k];
                    }
                }
                blurred[y * width + x] = (sum / weight.max(1)) as u8;
            }
        }
    }

    // Step 2: Compute gradients using Sobel
    // Use saturating_mul to prevent integer overflow
    let image_size = width.saturating_mul(height);
    let mut grad_x = vec![0f32; image_size];
    let mut grad_y = vec![0f32; image_size];
    let mut magnitude = vec![0f32; image_size];
    let mut direction = vec![0f32; image_size];

    const SOBEL_X: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const SOBEL_Y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;
                    let pixel = blurred[py * width + px] as i32;
                    gx += pixel * SOBEL_X[ky][kx];
                    gy += pixel * SOBEL_Y[ky][kx];
                }
            }

            let idx = y * width + x;
            grad_x[idx] = gx as f32;
            grad_y[idx] = gy as f32;
            magnitude[idx] = ((gx * gx + gy * gy) as f32).sqrt();
            direction[idx] = (gy as f32).atan2(gx as f32);
        }
    }

    // Step 3: Non-maximum suppression
    // Use saturating_mul to prevent integer overflow
    let image_size = width.saturating_mul(height);
    let mut suppressed = vec![0u8; image_size];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mag = magnitude[idx];
            let dir = direction[idx];

            // Quantize direction to 4 sectors (0, 45, 90, 135 degrees)
            let angle = ((dir + std::f32::consts::PI) * 4.0 / std::f32::consts::PI) as i32 % 4;

            let (dx1, dy1, dx2, dy2) = match angle {
                0 | 2 => (1, 0, -1, 0), // Horizontal
                1 => (1, 1, -1, -1),    // 45 degrees
                3 => (1, -1, -1, 1),    // 135 degrees
                _ => (0, 1, 0, -1),     // Vertical
            };

            let idx1 = ((y as isize + dy1) as usize) * width + ((x as isize + dx1) as usize);
            let idx2 = ((y as isize + dy2) as usize) * width + ((x as isize + dx2) as usize);

            let neighbor1 = magnitude[idx1];
            let neighbor2 = magnitude[idx2];

            if mag >= neighbor1 && mag >= neighbor2 {
                suppressed[idx] = mag.min(255.0) as u8;
            }
        }
    }

    // Step 4: Double threshold and hysteresis
    let mut dst_data = dst.data_mut();
    // Use saturating_mul to prevent integer overflow
    let image_size = width.saturating_mul(height);
    let mut edges = vec![0u8; image_size]; // 0=non-edge, 1=weak, 2=strong

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let val = suppressed[idx];

            if val >= high_threshold {
                edges[idx] = 2; // Strong edge
            } else if val >= low_threshold {
                edges[idx] = 1; // Weak edge
            }
        }
    }

    // Step 5: Edge tracking by hysteresis
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;

            if edges[idx] == 2 {
                // Strong edge - keep it
                dst_data[idx] = 255;
            } else if edges[idx] == 1 {
                // Weak edge - check if connected to strong edge
                let mut connected = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        let nidx = (ny as usize) * width + (nx as usize);
                        if edges[nidx] == 2 {
                            connected = true;
                            break;
                        }
                    }
                    if connected {
                        break;
                    }
                }

                if connected {
                    dst_data[idx] = 255;
                } else {
                    dst_data[idx] = 0;
                }
            } else {
                dst_data[idx] = 0;
            }
        }
    }

    Ok(())
}

/// Probabilistic Hough Transform for line detection
pub fn hough_lines_p(
    src: &Image,
    rho: f32,
    theta: f32,
    threshold: i32,
    min_line_length: i32,
    max_line_gap: i32,
) -> VxResult<Vec<LineSegment>> {
    let width = src.width() as i32;
    let height = src.height() as i32;

    // Validate input parameters to prevent division by zero or invalid dimensions
    if rho <= 0.0 || theta <= 0.0 || threshold <= 0 {
        return Ok(Vec::new());
    }

    // Hough accumulator dimensions
    let max_rho = ((width * width + height * height) as f32).sqrt();
    let rho_bins = (2.0 * max_rho / rho) as usize + 1;
    let theta_bins = (std::f32::consts::PI / theta) as usize + 1;

    // Ensure we have valid accumulator dimensions
    if rho_bins == 0 || theta_bins == 0 {
        return Ok(Vec::new());
    }

    // Accumulator
    let mut accumulator = vec![vec![0i32; theta_bins]; rho_bins];

    // Find edge points and vote
    let mut edge_points: Vec<(i32, i32)> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if src.get_pixel(x as usize, y as usize) > 128 {
                edge_points.push((x, y));

                // Vote in accumulator
                for t in 0..theta_bins {
                    let angle = t as f32 * theta;
                    let r = x as f32 * angle.cos() + y as f32 * angle.sin();
                    let r_idx = ((r + max_rho) / rho) as usize;
                    if r_idx < rho_bins {
                        accumulator[r_idx][t] += 1;
                    }
                }
            }
        }
    }

    // Find peaks in accumulator
    let mut lines = Vec::new();

    for r_idx in 0..rho_bins {
        for t_idx in 0..theta_bins {
            if accumulator[r_idx][t_idx] >= threshold {
                // Found a line
                let angle = t_idx as f32 * theta;
                let r = (r_idx as f32 * rho) - max_rho;

                // Find line segments
                let mut segment_points: Vec<(i32, i32)> = Vec::new();

                for &(x, y) in &edge_points {
                    let r_calc = x as f32 * angle.cos() + y as f32 * angle.sin();
                    if (r_calc - r).abs() < rho {
                        segment_points.push((x, y));
                    }
                }

                // Sort points along the line
                // Use unwrap_or to handle potential NaN values gracefully
                segment_points.sort_by(|a, b| {
                    let proj_a = a.0 as f32 * angle.cos() + a.1 as f32 * angle.sin();
                    let proj_b = b.0 as f32 * angle.cos() + b.1 as f32 * angle.sin();
                    proj_a.partial_cmp(&proj_b).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Extract line segments
                if segment_points.len() >= min_line_length as usize {
                    let mut start = segment_points[0];
                    let mut prev = start;

                    for i in 1..segment_points.len() {
                        let curr = segment_points[i];
                        let gap = ((curr.0 - prev.0).pow(2) + (curr.1 - prev.1).pow(2)) as f32;

                        if gap > (max_line_gap * max_line_gap) as f32 {
                            // End current segment
                            let dist =
                                ((curr.0 - start.0).pow(2) + (curr.1 - start.1).pow(2)) as f32;
                            if dist >= (min_line_length * min_line_length) as f32 {
                                lines.push(LineSegment::new(start.0, start.1, prev.0, prev.1));
                            }
                            start = curr;
                        }

                        prev = curr;
                    }

                    // Add last segment
                    let dist = ((prev.0 - start.0).pow(2) + (prev.1 - start.1).pow(2)) as f32;
                    if dist >= (min_line_length * min_line_length) as f32 {
                        lines.push(LineSegment::new(start.0, start.1, prev.0, prev.1));
                    }
                }
            }
        }
    }

    Ok(lines)
}

/// Threshold types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    Range,
}

/// Apply threshold to image
/// For BINARY: dst = (src > thresh) ? maxval : 0
/// For RANGE: dst = (src >= lower && src <= upper) ? maxval : 0
pub fn threshold(src: &Image, dst: &Image, thresh: u8, maxval: u8) -> VxResult<()> {
    threshold_binary(src, dst, thresh, maxval)
}

/// Binary threshold: dst = (src > thresh) ? maxval : 0
pub fn threshold_binary(src: &Image, dst: &Image, thresh: u8, maxval: u8) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let val = src.get_pixel(x, y);
            dst_data[y * width + x] = if val > thresh { maxval } else { 0 };
        }
    }

    Ok(())
}

/// Range threshold: dst = (src >= lower && src <= upper) ? maxval : 0
pub fn threshold_range(src: &Image, dst: &Image, lower: u8, upper: u8, maxval: u8) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let val = src.get_pixel(x, y);
            dst_data[y * width + x] = if val >= lower && val <= upper {
                maxval
            } else {
                0
            };
        }
    }

    Ok(())
}

/// Adaptive threshold using local mean
pub fn adaptive_threshold(src: &Image, dst: &Image, block_size: usize, c: i32) -> VxResult<()> {
    let width = src.width();
    let height = src.height();
    let half_block = (block_size / 2) as isize;
    let mut dst_data = dst.data_mut();

    for y in 0..height {
        for x in 0..width {
            let mut sum: u32 = 0;
            let mut count: u32 = 0;

            for dy in -half_block..=half_block {
                for dx in -half_block..=half_block {
                    let px = x as isize + dx;
                    let py = y as isize + dy;
                    if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                        sum += src.get_pixel(px as usize, py as usize) as u32;
                        count += 1;
                    }
                }
            }

            let mean = (sum / count.max(1)) as i32;
            let thresh = (mean - c).max(0).min(255) as u8;
            let val = src.get_pixel(x, y);

            dst_data[y * width + x] = if val > thresh { 255 } else { 0 };
        }
    }

    Ok(())
}
