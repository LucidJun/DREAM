

import yaml
import datetime
from importlib.resources import files


def generate_modelcard(config_file_path,output_path,version_num):

  with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

  """
    Generates an HTML model card report using a Jinja2 template and conditional rendering.

    Args:
        config_file_path (str): Path to the YAML-style config file.
        output_path (str): Path to save the generated HTML file.
        version_num (str): Version number of the model.
        uncert_desc (str, optional): Description of the uncertainty,
            if applicable. Defaults to None.

    Raises:
        Exception: If an error occurs during processing.
    """


  with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)



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


  # Dynamically resolve the paths to CSS files

  bootstrap_css_path = files('dreams_mc').joinpath('assets/vendor/bootstrap/css/bootstrap.min.css')
  font_awesome_css_path = files('dreams_mc').joinpath('assets/vendor/font-awesome/css/all.min.css')
  magnific_popup_css_path = files('dreams_mc').joinpath('assets/vendor/magnific-popup/magnific-popup.min.css')
  highlight_css_path = files('dreams_mc').joinpath('assets/vendor/highlight.js/styles/github.css')
  custom_stylesheet_path = files('dreams_mc').joinpath('assets/css/stylesheet.css')

 
  #Java script   
  jquery_path = files('dreams_mc').joinpath('assets/vendor/jquery/jquery.min.js')
  bootstrap_path = files('dreams_mc').joinpath('assets/vendor/bootstrap/js/bootstrap.bundle.min.js')
  highlight_js_path = files('dreams_mc').joinpath('assets/vendor/highlight.js/highlight.min.js')
  easing_path = files('dreams_mc').joinpath('assets/vendor/jquery.easing/jquery.easing.min.js')
  magnific_popup_path = files('dreams_mc').joinpath('assets/vendor/magnific-popup/jquery.magnific-popup.min.js')
  theme_js_path = files('dreams_mc').joinpath('assets/js/theme.js')


  try:
    #Define the HTML template
    model_card= f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1.0, shrink-to-fit=no">
    <link href="{config['logo_path']}" rel="icon" />
    <title>Overview | Your ThemeForest item Name</title>


    <!-- Stylesheet
    ============================== -->
    <!-- Bootstrap -->

    <link rel="stylesheet" type="text/css" href="{bootstrap_css_path}" />
    <!-- Font Awesome Icon -->
    <link rel="stylesheet" type="text/css" href="{font_awesome_css_path}" />
    <!-- Magnific Popup -->
    <link rel="stylesheet" type="text/css" href="{magnific_popup_css_path}" />
    <!-- Highlight Syntax -->
    <link rel="stylesheet" type="text/css" href="{highlight_css_path}" />
    <!-- Custom Stylesheet -->
    <link rel="stylesheet" type="text/css" href="{custom_stylesheet_path}" />
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

    <!-- Document Wrapper   
    =============================== -->
    <div id="main-wrapper"> 
      
      <!-- Header
      ============================ -->
      <header id="header" class="sticky-top"> 
        <!-- Navbar -->
        <nav class="primary-menu navbar navbar-expand-lg navbar-dropdown-dark">
          <div class="container-fluid">
            <!-- Sidebar Toggler -->
        <button id="sidebarCollapse" class="navbar-toggler d-block d-md-none" type="button"><span></span><span class="w-75"></span><span class="w-50"></span></button>
        
        <!-- Logo --> 
            <a class="logo ml-md-3" title="Logo"> <img src="{config['logo_path']}" alt="Logo"/> </a> 
            
             <div class="d-flex flex-column ml-md-2">
                <span class="text-2">{version_num}</span>
                
            </div>

          
            
        <!-- Navbar Toggler -->
        <button class="navbar-toggler ml-auto" type="button" data-toggle="collapse" data-target="#header-nav"><span></span><span></span><span></span></button>
            
        <div id="header-nav" class="collapse navbar-collapse justify-content-end">
              <ul class="navbar-nav">
              </ul>
           

          </div>
        </nav>
        <!-- Navbar End --> 
      </header>
      <!-- Header End --> 
      
      <!-- Content
      ============================ -->
      <div id="content" role="main">
        
      <!-- Sidebar Navigation
      ============================ -->
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
        
        <!-- Docs Content
      ============================ -->
        <div class="idocs-content">
          <div class="container"> 
            
          

        

        <!-- Getting Started
        ============================ -->
            <section id="overview">
            <h2 style="color: #5da834 ;" >Overview</h2>
            <p style="text-align: justify "> {config['describe_overview']} </p>
        
        
      
            
        <hr class="divider"> 

        

         <!-- Dataset
        ============================ -->
        <section id="dataset">
          <h2 style="color: #5da834 ;" >Dataset</h2>



          

        <ol>              
        <ul>

       	
        <li><strong>Dataset</strong> : {config['dataset_name']}</li>
        <li><strong>Number of Classes</strong> : {config['num_target_class']}</li>
        <li><strong>Ground Truth</strong> : {config['ground_truth']}</li>
        <li><strong>Training and Validation ratio</strong> : {config['split_ratio']}</li>
        <li><strong>Preprocessing</strong> : {config['preprocess_steps']}</li>
			  </li>
			  </ul>
        <br>
                
        </ol>

          <p style="text-align: justify;">{config['describe_dataset']}<br></p>


           <div style="flex: 1; padding: 10px;">
                    <a class="popup-img" href={config['data_figpath']}>
                        <img src={config['data_figpath']} class="img-fluid img-thumbnail" alt="image 1">
                    </a>
            </div>

        
    <hr class="divider">
        
            <!-- Model Details
        ============================ -->
            <section id="model_details">
              <h2 style="color: #5da834 ;" >Model Details</h2>
              
        <ol>
                
                
        <ul>

       	<li><strong>Input</strong>:{input_html}</li>
                
        <li><strong>Output</strong>:{output_html}</li>
       


        
        <li><strong>Learning Approach</strong> : {config['learning_approach']}</li>
        <li><strong>Model Type</strong> : {config['model_type']}</li>
        <li><strong>Learning Rate</strong> : {config['learning_rate']}</li>
        <li><strong>Batch Size</strong> : {config['batch_size']}</li>
        <li><strong>Additional Info</strong> : {config['additional_info']}</li>

			  </li>
			  </ul>
        <br>
        <p style="text-align: justify;">{config['model_details']}</p>
               
                
                
        </ol>
        
        </section>
            
        <hr class="divider">
        
            <!-- Performance
        ============================ -->
            <section id="performance">
              <h2 style="color: #5da834 ;" >Performance</h2>
              <p>{config['performance_comments']}<br></p>

                   
              <div style="display: flex; flex-direction: row;">

                <div style="flex: 1; padding: 10px;">
                    <a class="popup-img" href={config['result_table_figpath']}>
                        <img src={config['result_table_figpath']} class="img-fluid img-thumbnail" alt="image 1">
                    </a>
                    
                </div>

                <div style="flex: 1; padding: 10px;">
                    <a class="popup-img" href={config['cm_figpath']}>
                        <img src={config['cm_figpath']} class="img-fluid img-thumbnail" alt="image 2">
                    </a>
                </div>

              </div>

              <div style="display: flex; flex-direction: row;">

                <div style="flex: 1; padding: 10px;">
                    <a class="popup-img" href={config['acc_figpath']}>
                        <img src={config['acc_figpath']} class="img-fluid img-thumbnail" alt="image 1">
                    </a>
                </div>
                
                <div style="flex: 1; padding: 10px;">
                    <a class="popup-img" href={config['loss_figpath']}>
                        <img src={config['loss_figpath']} class="img-fluid img-thumbnail" alt="image 2">
                    </a>
                </div>
              </div>
          
            
        <hr class="divider">
                
        
        <!-- Image
        ============================ -->
            <section id="limitations">
              <h2 style="color: #5da834 ;" >Limitations</h2>
              <p class="text-4">{limitation_html}</p> 
              
         

        
        <hr class="divider">
        

                <!-- Uncertainity
        ============================ -->
        <section id="uncertainty">
          <h2 style="color: #5da834 ;" >Uncertainty</h2>
          <p style="text-align: justify;">{config['uncertainty_describe']}<br></p>



           <div style="flex: 1; padding: 10px;">
                    <a class="popup-img" href={config['uncertainty_figpath']}>
                        <img src={config['uncertainty_figpath']} class="img-fluid img-thumbnail" alt="image 1">
                    </a>
            </div>
            

        
    <hr class="divider">

              <!-- References
        ============================ -->
        <section id="references">
          <h2 style="color: #5da834 ;" >References</h2>
          <p> {config['references']}</p>

        
      <hr class="divider">
      
    </div>
    <!-- Document Wrapper end --> 

    <!-- Back To Top --> 
    <a id="back-to-top" data-toggle="tooltip" title="Back to Top" href="javascript:void(0)"><i class="fa fa-chevron-up"></i></a> 

    <!-- JavaScript
    ============================ -->


    <script src="{jquery_path}"></script>
    <script src="{bootstrap_path}"></script>
    <!-- Highlight JS -->
    <script src="{highlight_js_path}"></script>
    <!-- Easing -->
    <script src="{easing_path}"></script>
    <!-- Magnific Popup -->
    <script src="{magnific_popup_path}"></script>
    <!-- Custom Script -->
    <script src="{theme_js_path}"></script>
    </body>
    </html>

    '''
  # Write the HTML content to the output file
    with open(output_path, 'w') as output_file:
        output_file.write(model_card)
        
    print(f'HTML file created and saved at {output_path}')
  except Exception as e:
        print(f'An error occurred: {e}')




if __name__=="__main__" :

  #config = config.parse_arguments()
  generate_modelcard("/Users/rabindra/Developer/LucidJun/mc_config.yaml", "/Users/rabindra/Developer/LucidJun/model_card.html","V1.1")
