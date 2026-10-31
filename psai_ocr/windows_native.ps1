param(
    [string]$PdfPath,
    [int]$PageIndex = 0,
    [int]$Dpi = 180,
    [long]$MaxPixels = 20000000,
    [string]$LanguageTag = 'en-US',
    [string]$OutputImage,
    [Parameter(Mandatory=$true)][string]$ResultPath,
    [switch]$Probe
)

# Uses only Windows' installed PDF renderer/OCR language packs. No network,
# installation, policy changes or GUI. Confidence is unavailable in this API.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime]

$utf8 = New-Object System.Text.UTF8Encoding($false)
$languages = @([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object { $_.LanguageTag })
if ($Probe) {
    $result = [ordered]@{
        available = ($languages.Count -gt 0)
        languages = $languages
        engine = 'Windows.Media.Ocr'
        renderer = 'Windows.Data.Pdf'
        os_version = [Environment]::OSVersion.VersionString
        max_image_dimension = [Windows.Media.Ocr.OcrEngine]::MaxImageDimension
        confidence_available = $false
    }
    [IO.File]::WriteAllText($ResultPath, ($result | ConvertTo-Json -Depth 8), $utf8)
    exit 0
}

if (-not [IO.File]::Exists($PdfPath)) { throw 'PDF does not exist' }
if ($languages -notcontains $LanguageTag) { throw "OCR language pack is not installed: $LanguageTag" }
if ($PageIndex -lt 0 -or $Dpi -lt 72 -or $Dpi -gt 600 -or $MaxPixels -lt 10000) { throw 'Invalid page/render limits' }
if ([IO.File]::Exists($OutputImage) -or [IO.File]::Exists($ResultPath)) { throw 'Output already exists' }

$script:asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } |
    Select-Object -First 1
$script:asTaskAction = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object { $_.Name -eq 'AsTask' -and -not $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' } |
    Select-Object -First 1
function Wait-Operation($Operation, [Type]$ResultType) {
    $task = $script:asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}
function Wait-Action($Action) {
    $task = $script:asTaskAction.Invoke($null, @($Action))
    $task.Wait()
}

$started = [Diagnostics.Stopwatch]::StartNew()
$file = Wait-Operation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($PdfPath)) ([Windows.Storage.StorageFile])
$document = Wait-Operation ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
if ($PageIndex -ge $document.PageCount) { throw 'Page index is outside the PDF' }
$page = $document.GetPage([uint32]$PageIndex)
$stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
$bitmap = $null
$reader = $null
try {
    # WinRT page.Size is in DIPs (96/inch), unlike PDFium's PDF points (72/inch).
    $width = [double]$page.Size.Width * $Dpi / 96.0
    $height = [double]$page.Size.Height * $Dpi / 96.0
    $maximum = [double][Windows.Media.Ocr.OcrEngine]::MaxImageDimension
    $factor = [Math]::Min(1.0, [Math]::Min($maximum / $width, $maximum / $height))
    $factor = [Math]::Min($factor, [Math]::Sqrt($MaxPixels / ($width * $height)))
    $options = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $options.DestinationWidth = [uint32][Math]::Max(1, [Math]::Floor($width * $factor))
    $options.DestinationHeight = [uint32][Math]::Max(1, [Math]::Floor($height * $factor))
    Wait-Action ($page.RenderToStreamAsync($stream, $options))
    $stream.Seek(0)
    $decoder = Wait-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Wait-Operation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $language = New-Object Windows.Globalization.Language($LanguageTag)
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
    if ($null -eq $engine) { throw 'Windows could not create the OCR engine' }
    $recognized = Wait-Operation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $segments = New-Object System.Collections.Generic.List[object]
    $words = New-Object System.Collections.Generic.List[object]
    $lineIndex = 0
    foreach ($line in $recognized.Lines) {
        $left = [double]::PositiveInfinity
        $top = [double]::PositiveInfinity
        $right = 0.0
        $bottom = 0.0
        foreach ($word in $line.Words) {
            $rect = $word.BoundingRect
            $x = [double]$rect.X; $y = [double]$rect.Y
            $x2 = $x + [double]$rect.Width; $y2 = $y + [double]$rect.Height
            $left = [Math]::Min($left, $x); $top = [Math]::Min($top, $y)
            $right = [Math]::Max($right, $x2); $bottom = [Math]::Max($bottom, $y2)
            $words.Add([ordered]@{
                text = $word.Text; confidence = $null; segment_index = $lineIndex
                polygon = @(@($x, $y), @($x2, $y), @($x2, $y2), @($x, $y2))
            })
        }
        if ($line.Words.Count -gt 0) {
            $segments.Add([ordered]@{
                text = $line.Text; confidence = $null
                polygon = @(@($left, $top), @($right, $top), @($right, $bottom), @($left, $bottom))
            })
            $lineIndex += 1
        }
    }
    $stream.Seek(0)
    $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
    $loaded = Wait-Operation ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
    if ($loaded -ne $stream.Size) { throw 'Could not read the complete rendered page' }
    $bytes = New-Object byte[] ([int]$stream.Size)
    $reader.ReadBytes($bytes)
    [IO.File]::WriteAllBytes($OutputImage, $bytes)
    $result = [ordered]@{
        segments = @($segments.ToArray())
        words = @($words.ToArray())
        engine = [ordered]@{
            name = 'Windows.Media.Ocr'; version = [Environment]::OSVersion.VersionString
            device = 'cpu'; language = $LanguageTag; confidence_available = $false
            helper_execution_policy = 'process_only_Bypass'
            persistent_policy_changed = $false
            network_used = $false; model_downloaded = $false
            model_provenance = 'Windows installed OCR language capability; model weights not exposed by API'
            language_model_validation_complete = $false
        }
        renderer = 'Windows.Data.Pdf'
        renderer_version = [Environment]::OSVersion.VersionString
        elapsed_seconds = $started.Elapsed.TotalSeconds
        text_angle_degrees = $recognized.TextAngle
        width = $bitmap.PixelWidth; height = $bitmap.PixelHeight
    }
    [IO.File]::WriteAllText($ResultPath, ($result | ConvertTo-Json -Depth 16), $utf8)
}
finally {
    if ($null -ne $reader) { $reader.Dispose() }
    if ($null -ne $bitmap) { $bitmap.Dispose() }
    $stream.Dispose()
    $page.Dispose()
}
