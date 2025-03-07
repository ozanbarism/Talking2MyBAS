hvac_queries = {
    "Extracting time series of zone air temperature sensors and their corresponding VAVs": 
    """SELECT ?vav ?setpoint ?ts WHERE {
    ?vav rdf:type brick:VAV .
    ?setpoint brick:isPointOf ?vav. 
    ?setpoint ref:TimeseriesReference ?ts.
    ?setpoint rdf:type brick:Zone_Air_Temperature_Sensor .
    }""",
    
    "Extracting time series of zone air humidity sensors and their corresponding VAVs": 
    """SELECT ?vav ?hum ?ts WHERE {
    ?vav rdf:type brick:VAV .
    ?hum brick:isPointOf ?vav. 
    ?hum ref:TimeseriesReference ?ts.
    ?hum rdf:type brick:Zone_Air_Humidity_Sensor .
    }""",
    
    "Extracting time series of supply air flow sensors and their corresponding air handling units": 
    """SELECT ?ahu ?sup ?ts WHERE {
    ?ahu rdf:type brick:Air_Handling_Unit .
    ?sup brick:isPartOf ?ahu. 
    ?sup ref:TimeseriesReference ?ts.
    ?sup rdf:type brick:Supply_Air_Flow_Sensor .
    }""",
    
    "Extracting time series of damper position sensors and their corresponding VAVs, Air Handling Units, and Buildings": 
    """SELECT ?cell ?ahu ?vav ?damper ?ts WHERE {
    ?cell rdf:type brick:Building . 
    ?ahu rdf:type brick:Air_Handling_Unit .
    ?ahu brick:isPartOf ?cell .
    ?vav brick:isPartOf ?ahu .
    ?vav rdf:type brick:VAV .
    ?damper brick:isPointOf ?vav  .
    ?damper ref:TimeseriesReference ?ts.
    ?damper rdf:type brick:Damper_Position_Sensor.
    }""",
    
    "Extracting time series of damper position sensors": 
    """SELECT ?damper_pos ?ts WHERE {
    ?damper_pos ref:TimeseriesReference ?ts.
    ?damper_pos rdf:type brick:Damper_Position_Sensor.
    }""",

    "Extracting time series of supply air flow sensors and time series of return air flow sensors with their corresponding air handling units": """
    SELECT ?ahu ?sup_ts ?ret_ts
    WHERE {
        ?ahu rdf:type brick:Air_Handling_Unit .

        ?sup brick:isPartOf ?ahu .
        ?sup rdf:type brick:Supply_Air_Flow_Sensor .
        ?sup ref:TimeseriesReference ?sup_ts .

        ?ret brick:isPartOf ?ahu .
        ?ret rdf:type brick:Return_Air_Flow_Sensor .
        ?ret ref:TimeseriesReference ?ret_ts .
    }
    """,
    "Extracting time series of power sensors and their corresponding equipments, air handling units, and buildings": 
    """SELECT ?cell ?ahu ?eqp ?sensor ?ts WHERE {
    ?cell rdf:type brick:Building . 
    ?ahu rdf:type brick:Air_Handling_Unit .
    ?ahu brick:isPartOf ?cell .
    ?eqp brick:isPartOf ?ahu .
    ?sensor brick:isPointOf ?eqp  .
    ?sensor ref:TimeseriesReference ?ts.
    ?sensor rdf:type/rdfs:subClassOf* brick:Power_Sensor .
    }""",
    
    "Extracting time series of building electrical meters and their corresponding buildings": """
    SELECT ?building ?meter ?ts
    WHERE {
        ?building rdf:type brick:Building .
        ?meter brick:isPartOf ?building .
        ?meter rdf:type brick:Building_Electrical_Meter .
        ?meter ref:TimeseriesReference ?ts .
    }""",
    

    "Extracting time series of damper position sensors of return dampers": """
    SELECT ?damper ?time_series
    WHERE {
        ?damper rdf:type brick:Return_Damper .
        ?damper_pos brick:isPointOf ?damper. 
        ?damper_pos rdf:type brick:Damper_Position_Sensor .
        ?damper_pos ref:TimeseriesReference ?time_series .
    }""",
    
    "Extracting time series of zone air temperature sensors and time series of occupied cooling tempetaure setpoints and their corresponding VAVs.": """
    SELECT ?vav ?setpoint ?setpoint_ts ?temperature_sensor ?temperature_ts
    WHERE {
        ?vav rdf:type brick:VAV .
        
        # Occupied Cooling Setpoint
        ?setpoint brick:isPointOf ?vav .
        ?setpoint rdf:type brick:Occupied_Cooling_Temperature_Setpoint .
        ?setpoint ref:TimeseriesReference ?setpoint_ts .

        # Zone Air Temperature Sensor
        ?temperature_sensor brick:isPointOf ?vav .
        ?temperature_sensor rdf:type brick:Zone_Air_Temperature_Sensor .
        ?temperature_sensor ref:TimeseriesReference ?temperature_ts .
    }"""
}


