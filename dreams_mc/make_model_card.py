import os
import base64
import yaml
import pkg_resources
from importlib.resources import files
import datetime


def embed_file_content(file_path, tag_type):
    full_path = pkg_resources.resource_filename(__name__, file_path)
    with open(full_path, 'r') as file:
        content = file.read()
    if tag_type == 'css':
        return f'<style>\n{content}\n</style>'
    elif tag_type == 'js':
        return f'<script>\n{content}\n</script>'

def embed_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    mime_type = f"image/{os.path.splitext(image_path)[1][1:]}"
    return f'data:{mime_type};base64,{encoded_string}'

def generate_modelcard(config_file_path, output_path, version_num):
    import yaml
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Embedding CSS files
    bootstrap_css = embed_file_content('assets/vendor/bootstrap/css/bootstrap.min.css', 'css')
    font_awesome_css = embed_file_content('assets/vendor/font-awesome/css/all.min.css', 'css')
    magnific_popup_css = embed_file_content('assets/vendor/magnific-popup/magnific-popup.min.css', 'css')
    highlight_css = embed_file_content('assets/vendor/highlight.js/styles/github.css', 'css')
    custom_stylesheet = embed_file_content('assets/css/stylesheet.css', 'css')

    # Embedding JavaScript files
    jquery_js = embed_file_content('assets/vendor/jquery/jquery.min.js', 'js')
    bootstrap_js = embed_file_content('assets/vendor/bootstrap/js/bootstrap.bundle.min.js', 'js')
    highlight_js = embed_file_content('assets/vendor/highlight.js/highlight.min.js', 'js')
    easing_js = embed_file_content('assets/vendor/jquery.easing/jquery.easing.min.js', 'js')
    magnific_popup_js = embed_file_content('assets/vendor/magnific-popup/jquery.magnific-popup.min.js', 'js')
    theme_js = embed_file_content('assets/js/theme.js', 'js')

    # Embedding images
    logo_image = embed_image(config['logo_path'])
    data_image = embed_image(config['data_figpath'])
    result_table_image = embed_image(config['result_table_figpath'])
    cm_image = embed_image(config['cm_figpath'])
    acc_image = embed_image(config['acc_figpath'])
    loss_image = embed_image(config['loss_figpath'])
    uncertainty_image = embed_image(config['uncertainty_figpath'])

    output_list = config['model_output']
    output_html = "<ul>"
    for item in output_list:
        output_html += f"<li>{item}</li>"
    output_html += "</ul>"

    input_list = config['model_input']
    input_html = "<ul>"
    for item in input_list:
        input_html += f"<li>{item}</li>"
    input_html += "</ul>"

    limitations_list = config['limitation_details']
    limitation_html = "<ul>"
    for item in limitations_list:
        limitation_html += f"<li>{item}</li>"
    limitation_html += "</ul>"

    try:
        # Define the HTML template with inlined CSS and JavaScript
        model_card = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8" />
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1.0, shrink-to-fit=no">
        <link href="{logo_image}" rel="icon" />
        <title>Overview | Your ThemeForest item Name</title>

        <!-- Inline CSS -->
        {bootstrap_css}
        {font_awesome_css}
        {magnific_popup_css}
        {highlight_css}
        {custom_stylesheet}
        <style>
            .responsive-img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
        </head>

        <body data-spy="scroll" data-target=".idocs-navigation" data-offset="125">
        <!-- Preloader -->
        <div class="preloader">
            <div class="lds-ellipsis">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
        </div>
        <!-- Preloader End --> 

        <!-- Document Wrapper -->
        <div id="main-wrapper"> 
            <!-- Header -->
            <header id="header" class="sticky-top"> 
                <!-- Navbar -->
                <nav class="primary-menu navbar navbar-expand-lg navbar-dropdown-dark">
                    <div class="container-fluid">
                        <!-- Sidebar Toggler -->
                        <button id="sidebarCollapse" class="navbar-toggler d-block d-md-none" type="button"><span></span><span></span><span></span></button>

                        <!-- Logo --> 
                        <a class="logo ml-md-3" title="Logo"> <img src="{logo_image}" alt="Logo"/> </a> 
                        
                        <div class="d-flex flex-column ml-md-2">
                            <span class="text-2">{version_num}</span>
                        </div>

                        <!-- Navbar Toggler -->
                        <button class="navbar-toggler ml-auto" type="button" data-toggle="collapse" data-target="#header-nav"><span></span><span></span><span></span></button>
                        
                        <div id="header-nav" class="collapse navbar-collapse justify-content-end">
                            <ul class="navbar-nav">
                            </ul>
                        </div>
                    </div>
                </nav>
                <!-- Navbar End --> 
            </header>
            <!-- Header End --> 
            
            <!-- Content -->
            <div id="content" role="main">
                <!-- Sidebar Navigation -->
                <div class="idocs-navigation bg-light">
                    <ul class="nav flex-column "> 
                        <hr>
                        <li class="nav-item"><a class="nav-link" href="#overview">Overview</a></li>
                        <li class="nav-item"><a class="nav-link" href="#dataset">Dataset</a></li>
                        <li class="nav-item"><a class="nav-link" href="#model_details">Model Details</a></li>
                        <li class="nav-item"><a class="nav-link" href="#performance">Performance</a></li>
                        <li class="nav-item"><a class="nav-link" href="#limitations">Limitations</a></li>
                        <li class="nav-item"><a class="nav-link" href="#uncertainty">Uncertainty</a></li>
                        <li class="nav-item"><a class="nav-link" href="#references">References</a></li>
                    </ul>

                    <!-- Test Card -->
                    <div style="position: absolute; bottom: 0; left: 0; width: 100%;" class="text-center">
                        <span class="text-2" style="color: #07523e !important;">Date: {datetime.date.today()}</span>
                    </div>
                    <!-- Test Card End -->
                </div>

                <!-- Docs Content -->
                <div class="idocs-content">
                    <div class="container"> 
                        <!-- Getting Started -->
                        <section id="overview">
                            <h2 style="color: #5da834;">Overview</h2>
                            <p style="text-align: justify;">{config['describe_overview']}</p>
                        </section>

                        <hr class="divider"> 

                        <!-- Dataset -->
                        <section id="dataset">
                            <h2 style="color: #5da834;">Dataset</h2>
                            <ol>
                                <ul>
                                    <li><strong>Dataset</strong>: {config['dataset_name']}</li>
                                    <li><strong>Number of Classes</strong>: {config['num_target_class']}</li>
                                    <li><strong>Ground Truth</strong>: {config['ground_truth']}</li>
                                    <li><strong>Training and Validation ratio</strong>: {config['split_ratio']}</li>
                                    <li><strong>Preprocessing</strong>: {config['preprocess_steps']}</li>
                                </ul>
                            </ol>
                            <br>
                            <p style="text-align: justify;">{config['describe_dataset']}</p>
                            <div style="flex: 1; padding: 10px;">
                                <a class="popup-img" href="{data_image}">
                                    <img src="{data_image}" class="img-fluid img-thumbnail responsive-img" alt="image 1">
                                </a>
                            </div>
                        </section>

                        <hr class="divider">

                        <!-- Model Details -->
                        <section id="model_details">
                            <h2 style="color: #5da834;">Model Details</h2>
                            <ol>
                                <ul>
                                    <li><strong>Input</strong>: {input_html}</li>
                                    <li><strong>Output</strong>: {output_html}</li>
                                    <li><strong>Learning Approach</strong>: {config['learning_approach']}</li>
                                    <li><strong>Model Type</strong>: {config['model_type']}</li>
                                    <li><strong>Learning Rate</strong>: {config['learning_rate']}</li>
                                    <li><strong>Batch Size</strong>: {config['batch_size']}</li>
                                    <li><strong>Additional Info</strong>: {config['additional_info']}</li>
                                </ul>
                            </ol>
                            <br>
                            <p style="text-align: justify;">{config['model_details']}</p>
                        </section>

                        <hr class="divider">

                        <!-- Performance -->
                        <section id="performance">
                            <h2 style="color: #5da834;">Performance</h2>
                            <p>{config['performance_comments']}<br></p>
                            <div style="display: flex; flex-direction: row;">
                                <div style="flex: 1; padding: 10px;">
                                    <a class="popup-img" href="{result_table_image}">
                                        <img src="{result_table_image}" class="img-fluid img-thumbnail responsive-img" alt="image 1">
                                    </a>
                                </div>
                                <div style="flex: 1; padding: 10px;">
                                    <a class="popup-img" href="{cm_image}">
                                        <img src="{cm_image}" class="img-fluid img-thumbnail responsive-img" alt="image 2">
                                    </a>
                                </div>
                            </div>
                            <div style="display: flex; flex-direction: row;">
                                <div style="flex: 1; padding: 10px;">
                                    <a class="popup-img" href="{acc_image}">
                                        <img src="{acc_image}" class="img-fluid img-thumbnail responsive-img" alt="image 1">
                                    </a>
                                </div>
                                <div style="flex: 1; padding: 10px;">
                                    <a class="popup-img" href="{loss_image}">
                                        <img src="{loss_image}" class="img-fluid img-thumbnail responsive-img" alt="image 2">
                                    </a>
                                </div>
                            </div>
                        </section>

                        <hr class="divider">

                        <!-- Limitations -->
                        <section id="limitations">
                            <h2 style="color: #5da834;">Limitations</h2>
                            <p class="text-4">{limitation_html}</p>
                        </section>

                        <hr class="divider">

                        <!-- Uncertainty -->
                        <section id="uncertainty">
                            <h2 style="color: #5da834;">Uncertainty</h2>
                            <p style="text-align: justify;">{config['uncertainty_describe']}<br></p>
                            <div style="flex: 1; padding: 10px;">
                                <a class="popup-img" href="{uncertainty_image}">
                                    <img src="{uncertainty_image}" class="img-fluid img-thumbnail responsive-img" alt="image 1">
                                </a>
                            </div>
                        </section>

                        <hr class="divider">

                        <!-- References -->
                        <section id="references">
                            <h2 style="color: #5da834;">References</h2>
                            <p>{config['references']}</p>
                        </section>

                        <hr class="divider">
                    </div>
                </div>
            </div>
        </div>
        <!-- Document Wrapper end --> 

        <!-- Back To Top --> 
        <a id="back-to-top" data-toggle="tooltip" title="Back to Top" href="javascript:void(0)"><i class="fa fa-chevron-up"></i></a> 

        <!-- Inline JavaScript -->
        {jquery_js}
        {bootstrap_js}
        {highlight_js}
        {easing_js}
        {magnific_popup_js}
        {theme_js}
        </body>
        </html>
        '''
        # Write the HTML content to the output file
        with open(output_path, 'w') as output_file:
            output_file.write(model_card)
        
        print(f'HTML file created and saved at {output_path}')
    except Exception as e:
        print(f'An error occurred: {e}')

# Example usage
if __name__ == "__main__":
  generate_modelcard("/Users/rabindra/Developer/LucidJun/mc_config.yaml", "/Users/rabindra/Developer/LucidJun/model_card_inline.html","V1.1")
